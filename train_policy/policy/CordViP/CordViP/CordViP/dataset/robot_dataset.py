from typing import Dict
import torch
import numpy as np
import copy
from CordViP.common.pytorch_util import dict_apply
from CordViP.common.replay_buffer import ReplayBuffer
from CordViP.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from CordViP.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from CordViP.dataset.base_dataset import BaseDataset
import pdb

class RobotDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        print ("zarr_path:",zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['joint_state', 'joint_action', 'hand_point_cloud', 'object_point_cloud', 'contact_map']) # 'img'
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'joint_action': self.replay_buffer['joint_action'],
            'joint_state': self.replay_buffer['joint_state'][...,:],
            'hand_point_cloud': self.replay_buffer['hand_point_cloud'],
            'object_point_cloud': self.replay_buffer['object_point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        joint_state = sample['joint_state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        hand_point_cloud = sample['hand_point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        object_point_cloud = sample['object_point_cloud'][:,].astype(np.float32)

        data = {
            'obs': {
                'hand_point_cloud': hand_point_cloud,
                'object_point_cloud': object_point_cloud, # T, 1024, 3
                'joint_state': joint_state, # T, D_pos
            },
            'joint_action': sample['joint_action'].astype(np.float32) ,# T, D_action
            'contact_map': sample['contact_map'].astype(np.float32) 
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == '__main__':
    test = RobotDataset('../../data/ctx.zarr')
    print('ready')