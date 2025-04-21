import pdb, pickle, os
import numpy as np
import open3d as o3d
from copy import deepcopy
import zarr, shutil
import argparse
import torch
import pytorch3d.ops as torch3d_ops
import open3d as o3d
from typing import Dict
import pickle
import os


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    parser.add_argument('episode_number', type=int)

    args = parser.parse_args()

    task_name = args.task_name
    num = args.episode_number
    current_ep = 0
    
    load_dir = f'/home/fqx/expert_dataset_new/{task_name}/recorded_data'
    
    total_count = 0

    save_dir = f'./policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/{task_name}_{num}.zarr'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    point_cloud_arrays, episode_ends_arrays, joint_state_arrays, joint_action_arrays = [], [], [], []

    num_files = 300
    while os.path.isdir(load_dir+f'/demonstration_{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        point_cloud_sub_arrays = []
        joint_state_sub_arrays = []
        joint_action_sub_arrays = []

        # 5hz
        while os.path.exists(load_dir+f'/demonstration_{current_ep}'+f'/{file_num}') and file_num < num_files:
            with open(load_dir+f'/demonstration_{current_ep}'+f'/{file_num}', 'rb') as file:
                data = pickle.load(file)

            point_cloud = data['point_cloud_1024'][:,:]
            
            hand_joint_positions = np.array(data['hand_joint_positions'])  
            arm_joint_positions = np.array(data['arm_joint_positions'])    
            joint_state = np.concatenate([hand_joint_positions, arm_joint_positions])
            
            if file_num + 1 < num_files:
                next_data = pickle.load(open(load_dir + f'/demonstration_{current_ep}' + f'/{file_num + 2}', 'rb'))
                hand_commanded_joint_position = np.array(next_data['hand_joint_positions'])
                arm_commanded_joint_position = np.array(next_data['arm_joint_positions'])
            else:
                hand_commanded_joint_position = np.array(data['hand_joint_positions'])
                arm_commanded_joint_position = np.array(data['arm_joint_positions'])

            joint_action = np.concatenate([hand_commanded_joint_position, arm_commanded_joint_position])

            point_cloud_sub_arrays.append(point_cloud)
            joint_state_sub_arrays.append(joint_state)
            joint_action_sub_arrays.append(joint_action)

            total_count += 1
            file_num += 1
        
        current_ep += 1

        episode_ends_arrays.append(deepcopy(total_count))
        point_cloud_arrays.extend(point_cloud_sub_arrays)
        joint_state_arrays.extend(joint_state_sub_arrays)
        joint_action_arrays.extend(joint_action_sub_arrays)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    point_cloud_arrays = np.array(point_cloud_arrays)
    joint_state_arrays = np.array(joint_state_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    joint_state_chunk_size = (100, joint_state_arrays.shape[1])
    joint_action_chunk_size = (100, joint_action_arrays.shape[1])


    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('joint_state', data=joint_state_arrays, chunks=joint_state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('joint_action', data=joint_action_arrays, chunks=joint_action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()
