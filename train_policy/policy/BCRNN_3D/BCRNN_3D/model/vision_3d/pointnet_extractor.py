import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from BCRNN_3D.model.vision_3d.positional_encoding import PositionalEncoding
from BCRNN_3D.model.vision_3d.pointnet2_encoder import PointNet2EncoderXYZ
from BCRNN_3D.model.vision_3d.pointnext_encoder import PointNextEncoderXYZ
try:
    from BCRNN_3D.model.vision_3d.voxelcnn_encoder import VoxelCNN
except:
    print("voxel cnn not found. pass")
from BCRNN_3D.model.vision_3d.pointtransformer_encoder import Backbone as PT_Backbone

from matplotlib import pyplot as plt

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, simnorm_dim=8):
		super().__init__()
		self.dim = simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_positional_encoding: bool=False,
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        if final_norm == 'simnorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                SimNorm(),
            )
        elif final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_positional_encoding: bool=False,
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_in=in_channels, include_input=True)
            in_channels = self.positional_encoding.d_out
            cprint(f"[PointNetEncoderXYZ] use positional encoding, in_channels: {in_channels}", "green")
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        if final_norm == 'simnorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                SimNorm(),
            )
        elif final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
         
         
    def forward(self, x):
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    
class PointTransformer(nn.Module):
    """Encoder for pointtransformer
    """
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 normal_channel: bool=False, 
                 final_norm: str='none',
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            out_channels (int): feature size of output
            normal_channel (bool): whether to use RGB. Defaults to False.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normal_channel = normal_channel
        
        if self.normal_channel:
            assert in_channels == 6, cprint(f"PointTransformer only supports 6 channels, but got {in_channels}", "red")
        else:
            assert in_channels == 3, cprint(f"PointTransformer only supports 3 channels, but got {in_channels}", "red")
        #self.transformer = PT_Backbone(512,4,16,128,self.in_channels)
        
        self.transformer = PT_Backbone(512,4,512,128,self.in_channels)#number of points, number of blocks, number of nearest neighbors(knn), transformer hidden dimension, dimension of points(default=3)
        if final_norm == 'simnorm':
            self.final_projection = nn.Sequential(
                nn.Linear(512, out_channels),
                SimNorm(),
            )
        elif final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(512, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(512, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        
    def forward(self, x):
        features, _ = self.transformer(x)
        features = torch.max(self.final_projection(features),dim=1)[0]
        return features


class PointNetImaginationExtractor(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 use_RGB=False,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.point_cloud_robot_key = 'point_cloud_robot'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.use_point_cloud_robot = self.point_cloud_robot_key in observation_space.keys()
        self.use_rgb_image = self.rgb_image_key in observation_space.keys() and use_RGB
        # self.use_rgb_image = use_RGB
        cprint("[PointNetImaginationExtractor] use_rgb_image: {}".format(self.use_rgb_image), "yellow")

        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        if self.use_rgb_image:
            self.rgb_image_shape = observation_space[self.rgb_image_key]
            # this is not good
            # self.rgb_encoder = torchvision.models.resnet18(pretrained=True)
            # # remove the last classification layer and replace with a squeeze layer
            # self.rgb_encoder = nn.Sequential(*list(self.rgb_encoder.children())[:-1], nn.Flatten())
            # self.rgb_feat_dim = 512
            
            from robomimic.algo import algo_factory
            from robomimic.algo.algo import PolicyAlgo
            import robomimic.utils.obs_utils as ObsUtils
            import robomimic.models.base_nets as rmbn
            from BCRNN_3D.common.robomimic_config_util import get_robomimic_config
            import BCRNN_3D.model.vision.crop_randomizer as dmvc
            from BCRNN_3D.common.pytorch_util import dict_apply, replace_submodules
            
            obs_config = {
                'low_dim': [],
                'rgb': ['image'],
                'depth': [],
                'scan': []
            }
            config = get_robomimic_config(
                algo_name='bc_rnn',
                hdf5_type='image',
                task_name='square',
                dataset_type='ph')
            
            with config.unlocked():
                # set config with shape_meta
                config.observation.modalities.obs = obs_config
                if img_crop_shape is None:
                    for key, modality in config.observation.encoder.items():
                        if modality.obs_randomizer_class == 'CropRandomizer':
                            modality['obs_randomizer_class'] = None
                else:
                    # set random crop parameter
                    ch, cw = img_crop_shape
                    for key, modality in config.observation.encoder.items():
                        if modality.obs_randomizer_class == 'CropRandomizer':
                            modality.obs_randomizer_kwargs.crop_height = ch
                            modality.obs_randomizer_kwargs.crop_width = cw
            
            # init global state
            ObsUtils.initialize_obs_utils_with_config(config)

            # load model
            policy: PolicyAlgo = algo_factory(
                    algo_name=config.algo_name,
                    config=config,
                    obs_key_shapes={'image': self.rgb_image_shape},
                    ac_dim=30, # this param is not used, so it doesn't matter
                    device='cpu',
                )
            
            obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
            obs_encoder_group_norm = True
            if obs_encoder_group_norm:
                # replace batch norm with group norm
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features//16, 
                        num_channels=x.num_features)
                )
                # obs_encoder.obs_nets['agentview_image'].nets[0].nets
            
            # obs_encoder.obs_randomizers['agentview_image']
            eval_fixed_crop = True
            if eval_fixed_crop:
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                    func=lambda x: dmvc.CropRandomizer(
                        input_shape=x.input_shape,
                        crop_height=x.crop_height,
                        crop_width=x.crop_width,
                        num_crops=x.num_crops,
                        pos_enc=x.pos_enc
                    )
                )

            self.rgb_feat_dim = obs_encoder.output_shape()[0]
            self.rgb_encoder = obs_encoder
            self.n_output_channels += self.rgb_feat_dim
        else:
            self.rgb_image_shape = None
        
        cprint(f"[PointNetImaginationExtractor] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[PointNetImaginationExtractor] state shape: {self.state_shape}", "yellow")
        cprint(f"[PointNetImaginationExtractor] imagination point shape: {self.imagination_shape}", "yellow")
        cprint(f"[PointNetImaginationExtractor] rgb image shape: {self.rgb_image_shape}", "yellow")
        

        # not used
        from BCRNN_3D.model.vision_3d.pointnet import PointNet, PointNetMedium, PointNetLarge
        
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointnet2":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                pointcloud_encoder_cfg.normal_channel = True
                self.extractor = PointNet2EncoderXYZ(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                pointcloud_encoder_cfg.normal_channel = False
                self.extractor = PointNet2EncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointnext":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                pointcloud_encoder_cfg.normal_channel = True
                self.extractor = PointNextEncoderXYZ(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                pointcloud_encoder_cfg.normal_channel = False
                self.extractor = PointNextEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointtransformer":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                pointcloud_encoder_cfg.normal_channel = True
                self.extractor = PointTransformer(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                pointcloud_encoder_cfg.normal_channel = False
                self.extractor = PointTransformer(**pointcloud_encoder_cfg)
        elif pointnet_type == "voxelcnn":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = VoxelCNN(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = VoxelCNN(**pointcloud_encoder_cfg)
        elif "openpoints" in pointnet_type:
            from .openpoints_encoder import OpenPointsEncoder
            pointnet_type = pointnet_type.split("_")[-1]
            self.extractor = OpenPointsEncoder(pointnet_type=pointnet_type, **pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        
        if self.use_point_cloud_robot:
            self.extractor_robot = copy.deepcopy(self.extractor)
            self.n_output_channels += out_channel

        cprint(f"[PointNetImaginationExtractor] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel

        if self.use_rgb_image:
            rgb_image = observations[self.rgb_image_key]
            rgb_feat = self.rgb_encoder({'image': rgb_image})
            pn_feat = torch.cat([pn_feat, rgb_feat], dim=-1)
        
        if self.use_point_cloud_robot:
            points_robot = observations[self.point_cloud_robot_key]
            pn_feat_robot = self.extractor_robot(points_robot)
            pn_feat = torch.cat([pn_feat, pn_feat_robot], dim=-1)
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels