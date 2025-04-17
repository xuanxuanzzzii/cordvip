from typing import Dict
import torch
from BCRNN_3D.model.common.normalizer import LinearNormalizer
from BCRNN_3D.policy.base_image_policy import BaseImagePolicy
from BCRNN_3D.common.pytorch_util import dict_apply

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from BCRNN_3D.common.robomimic_config_util import get_robomimic_config
from BCRNN_3D.model.vision_3d.pointnet_extractor import PointNetEncoderXYZ

class RobomimicPointcloudPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            algo_name='bc_rnn',
            obs_type='pointcloud',
            task_name='square',
            dataset_type='ph',
            crop_shape=(76,76),
            use_RGB=False,
            encoder_output_dim=50,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
        ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['joint_action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        
        obs_config = {
                'low_dim': ['joint_state', 'point_cloud'],
                'rgb': ['image'],
                'depth': [],
                'scan': []
            }
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # load model
        low_dim_shape = obs_shape_meta['joint_state']['shape'][0] + encoder_output_dim
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes={
                    # 'image': [3,84,84],
                    # 'point_cloud': obs_shape_meta['point_cloud']['shape'],
                    'point_cloud': [encoder_output_dim],
                    # 'low_dim': [low_dim_shape],
                    'low_dim': obs_shape_meta['joint_state']['shape'],
                    },
                ac_dim=action_dim,
                device='cuda',
            )

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        

        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.config = config
        
        

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_dict = self.normalizer(obs_dict)
        if not self.use_pc_color:
            nobs_dict['point_cloud'] = nobs_dict['point_cloud'][...,:3]
            
        point_cloud = nobs_dict['point_cloud']
        B, T, N, C = point_cloud.shape
        point_cloud_latent = self.extractor(point_cloud.reshape(B*T, N, C)).reshape(B, T, -1)
        nobs_dict['point_cloud'] = point_cloud_latent
        
        robomimic_obs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['joint_action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result

    def reset(self):
        self.model.reset()

    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_on_batch(self, batch, epoch, validate=False):
        nobs = self.normalizer.normalize(batch['obs'])
        point_cloud = nobs['point_cloud']
        if not self.use_pc_color:
            point_cloud = point_cloud[...,:3]
        B, T, N, C = point_cloud.shape
        point_cloud_latent = self.extractor(point_cloud.reshape(B*T, N, C)).reshape(B, T, -1)
        nobs['point_cloud'] = point_cloud_latent
        nactions = self.normalizer['joint_action'].normalize(batch['joint_action'])
        robomimic_batch = {
            'obs': nobs,
            'actions': nactions
        }
        input_batch = self.model.process_batch_for_training(
            robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info
    
    def on_epoch_end(self, epoch):
        self.model.on_epoch_end(epoch)

    def get_optimizer(self):
        return self.model.optimizers['policy']

