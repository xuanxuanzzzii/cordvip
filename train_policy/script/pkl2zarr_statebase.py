import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import einops
import cv2
import math


def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='pick_and_place',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('episode_number', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    num = args.episode_number
    
    load_dir = f'./expert_dataset/{task_name}/recorded_data'
    total_count = 0

    save_dir = f'./policy/statebase_Diffusion-Policy/data/{task_name}_{num}.zarr'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    img_arrays = []
    episode_ends_arrays, joint_state_arrays, joint_action_arrays = [], [], []
    
    num_files = 500
    while os.path.isdir(load_dir+f'/demonstration_{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        
        while os.path.exists(load_dir+f'/demonstration_{current_ep}'+f'/{file_num}') and file_num < num_files:
            with open(load_dir+f'/demonstration_{current_ep}'+f'/{file_num}', 'rb') as file:
                data = pickle.load(file)
            
            img = data['camera_1_color_use_image']
            img = img[:, : ,[2,1,0]] 
            
            hand_joint_positions = np.array(data['hand_joint_positions'])  
            arm_joint_positions = np.array(data['arm_joint_positions']) 
            # print(data.keys())
            pos_6d = np.array(data['pose']) 
            pos_6d_2 = np.array(data['pose_2']) 
            joint_state = np.concatenate([hand_joint_positions, arm_joint_positions, pos_6d, pos_6d_2])

            if file_num + 1 < num_files:
                next_data = pickle.load(open(load_dir + f'/demonstration_{current_ep}' + f'/{file_num + 2}', 'rb'))
                hand_commanded_joint_position = np.array(next_data['hand_joint_positions'])
                arm_commanded = np.array(next_data['arm_joint_positions'])
            else:
                hand_commanded_joint_position = np.array(data['hand_joint_positions'])
                arm_commanded = np.array(data['arm_joint_positions'])

            joint_action = np.concatenate([hand_commanded_joint_position, arm_commanded])
            
            img_arrays.append(img)
            joint_state_arrays.append(joint_state)
            joint_action_arrays.append(joint_action)
            
            total_count += 1

            file_num += 1

        current_ep += 1

        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    joint_state_arrays = np.array(joint_state_arrays)
    img_arrays = np.array(img_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
     
    img_arrays = np.moveaxis(img_arrays, -1, 1)  

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    joint_state_chunk_size = (100, joint_state_arrays.shape[1])
    joint_action_chunk_size = (100, joint_action_arrays.shape[1])
    img_chunk_size = (100, *img_arrays.shape[1:])
    # zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('joint_state', data=joint_state_arrays, chunks=joint_state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('joint_action', data=joint_action_arrays, chunks=joint_action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()
