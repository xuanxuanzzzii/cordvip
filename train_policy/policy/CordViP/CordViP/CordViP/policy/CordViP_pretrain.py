from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pdb

from CordViP.model.common.normalizer import LinearNormalizer
from CordViP.policy.base_policy import BasePolicy
from CordViP.model.diffusion.conditional_unet1d import ConditionalUnet1D
from CordViP.model.diffusion.mask_generator import LowdimMaskGenerator
from CordViP.common.pytorch_util import dict_apply
from CordViP.common.model_util import print_params
from CordViP.model.vision.transformer import Transformer
from CordViP.model.vision.encoder import PointROEncoder
from CordViP.model.vision.create_fc import create_fc

class CordViP_Pretrain(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            encoder_emb_dim=1024,
            encoder_pretrain=None,
            robot_embedding_dim=64,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()
        
        print("horizon:",horizon)
        print("n_obs_steps",n_obs_steps)
        print("n_action_steps",n_action_steps)
        self.condition_type = condition_type

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = PointROEncoder(observation_space=obs_dict,
                                    encoder_emb_dim=encoder_emb_dim,
                                    encoder_pretrain=encoder_pretrain,
                                    use_pc_color=use_pc_color,
                                    pointnet_type=pointnet_type,
                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                    robot_embedding_dim=robot_embedding_dim,
                                    )

        self.obs_encoder = obs_encoder
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        
        self.robot_embedding_dim = robot_embedding_dim
        self.linear_input_dim = (encoder_emb_dim * 2 + self.robot_embedding_dim * 1) * self.n_obs_steps
        self.linear_layer_pred_arm = create_fc(input_dim=self.linear_input_dim,output_dim=6,hidden_dims=[512,256,64])
        self.linear_layer_pred_hand = create_fc(input_dim=self.linear_input_dim,output_dim=16,hidden_dims=[512,256,64])
        self.predict_contact_map = create_fc(input_dim=encoder_emb_dim*2,output_dim=encoder_emb_dim,hidden_dims=[2048,1024,1024])
        
        print_params(self)
        print("init finished")

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss_hand_and_arm(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])

        nactions = self.normalizer['joint_action'].normalize(batch['joint_action'])
        batch_size = nactions.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...])
        robot_embedding_tf, object_embedding_tf, hand_embedding_con, arm_embedding_con = self.obs_encoder(this_nobs)

        # contact map
        contact_map_gt = batch['contact_map']
        contact_map_gt = contact_map_gt[:, :self.n_obs_steps, :]
        
        combined_input = torch.cat((robot_embedding_tf, object_embedding_tf), dim=-1)
        contact_map_predict = self.predict_contact_map(combined_input)

        loss_contact_map = F.mse_loss(contact_map_predict,contact_map_gt,reduction='none')
        loss_contact_map = reduce(loss_contact_map, 'b ... -> b (...)', 'mean')
        loss_contact_map = loss_contact_map.mean()

        combined_hand_embedding = torch.cat((robot_embedding_tf.reshape(batch_size, -1),
                                            object_embedding_tf.reshape(batch_size, -1),
                                            hand_embedding_con.reshape(batch_size, -1)), dim=1)
        
        combined_arm_embedding = torch.cat((robot_embedding_tf.reshape(batch_size, -1),
                                            object_embedding_tf.reshape(batch_size, -1),
                                            arm_embedding_con.reshape(batch_size, -1)), dim=1)
        

        arm_pred = self.linear_layer_pred_arm(combined_hand_embedding) #(32,6)
        hand_pred = self.linear_layer_pred_hand(combined_arm_embedding)
        

        joint_state_data = nobs['joint_state'] # hand:16 arm:6 torch.Size([32, 12, 22])
        joint_state_at_t4 = joint_state_data[:, self.n_obs_steps-1, :] #(32,22)
        joint_state_at_t4_hand_gt = joint_state_at_t4[:, :16] #(32,16)
        joint_state_at_t4_arm_gt = joint_state_at_t4[:, 16:] #(32,6)

        loss_hand = F.mse_loss(hand_pred,joint_state_at_t4_hand_gt,reduction='none') 
        loss_hand = reduce(loss_hand, 'b ... -> b (...)', 'mean')
        loss_hand = loss_hand.mean()

        loss_arm = F.mse_loss(arm_pred,joint_state_at_t4_arm_gt,reduction='none')
        loss_arm = reduce(loss_arm, 'b ... -> b (...)', 'mean')
        loss_arm = loss_arm.mean()

        loss = loss_arm + loss_hand + loss_contact_map
        return loss