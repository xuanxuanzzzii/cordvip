import os
import sys
import argparse
import time
# import viser
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
import open3d as o3d
import pickle

def load_and_visualize_pkl_files(directory_path,  num_points=1024,  save_dir=None):
    """Reads all .pkl files in the specified directory, loads q, and visualizes the sampled point cloud."""
    pkl_files = [f for f in os.listdir(directory_path) if f.isdigit()]  

    pkl_files.sort(key=int)
    hand = create_hand_model(args.robot_name, torch.device('cuda'), args.num_points)

    for idx, pkl_file in enumerate(pkl_files, start=1):
        time_1 = time.time()
        file_path = os.path.join(directory_path, pkl_file)
        print(f"Processing {file_path}...")
        # hand = create_hand_model(args.robot_name, torch.device('cpu'), args.num_points)
 
        q = load_single_demonstration(file_path)
        q = q[[0,1,2,3,4,5,7,6,8,9,11,10,12,13,15,14,16,17,18,19,20,21]]
        sampled_pc, sampled_pc_index, sampled_transform = hand.get_sampled_pc(q=q, num_points=num_points)
        print("sampled_pc:", sampled_pc)
        if save_dir:
            save_path = os.path.join(save_dir, f"{idx}.pt")
            data = {
                'sample': sampled_pc,
                'transform':sampled_transform
            }
            torch.save(data, save_path)
            print(f"Saved {save_path}")
            
        time_2 = time.time()
        print('elaspe',1/(time_2 - time_1))
        


def load_single_demonstration(pkl_file):
    """Load a single TCP and joint data from a .pkl file."""
    with open(pkl_file, 'rb') as f:
        demonstration_data = pickle.load(f)
    
    # Extract the tcp and joint data from the loaded file
    tcp = torch.tensor(demonstration_data['arm_joint_positions'], dtype=torch.float32)
    joint = torch.tensor(demonstration_data['hand_joint_positions'], dtype=torch.float32)
    
    # Combine tcp and joint to form q (or use one of them as per your requirements)
    q = torch.cat([tcp, joint], dim=0)  # Assuming concatenation is appropriate
    
    return q



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='robot', type=str)
    parser.add_argument('--save_path', default='data/PointCloud/demonstration/', type=str)
    parser.add_argument('--num_points', default=1024, type=int)
    parser.add_argument('--object_source_path', default='data/data_urdf/object', type=str)
    parser.add_argument('--robot_name', default='leaphand', type=str)
    parser.add_argument('--demonstration_folder_path', type=str, required=True)
    parser.add_argument('--output_cloud_dir', type=str, required=True)
    args = parser.parse_args()
    demonstration_folder_path = args.demonstration_folder_path
    output_dir = args.output_cloud_dir
    os.makedirs(output_dir, exist_ok=True)

    load_and_visualize_pkl_files(demonstration_folder_path,  args.num_points, save_dir=output_dir)
