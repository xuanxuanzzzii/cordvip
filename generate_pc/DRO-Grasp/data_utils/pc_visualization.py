import os
import torch
import numpy as np
import open3d as o3d
import time
import pickle
import copy
import argparse


def load_point_cloud(pt_file_path):
    """Load the point cloud from a .pt file."""
    data = torch.load(pt_file_path)
    sampled_pc = data['sample']  
    tansform_pc = data['transform']
    return sampled_pc[:, :3].numpy() ,tansform_pc 

def load_pose_data(pose_folder_path, num_frames=500):
    """
    Load the object pose data from .pkl files in the pose folder.
    """
    poses = []
    object_poses = []
    for i in range(num_frames):
        pkl_file = os.path.join(pose_folder_path, f'{i}')
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            object_pose = data['object_pose']

            translation = object_pose[:3, 3]
            rotation = object_pose[:3, :3]

            poses.append({"translation": translation, "rotation": rotation})
            object_poses.append(object_pose)
            
    return poses , object_poses

def save_data_to_file(save_folder, file_counter, object_pose, hand_cloud, object_cloud):
    """
    Save the data to a .pkl file. If the file already exists, update its dictionary with the new data.
    """
    file_name = f"{file_counter}"
    file_path = os.path.join(save_folder, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded existing data from {file_path}")
    else:
        data = {'object_pose': [], 'hand_cloud': [], 'object_cloud': []}
        print(f"Created new data file {file_path}")
        
    if 'object_pose' not in data:
        data['object_pose'] = []
    if 'hand_cloud' not in data:
        data['hand_cloud'] = []
    if 'object_cloud' not in data:
        data['object_cloud'] = []

    data['object_pose'] = object_pose
    data['hand_cloud'] = hand_cloud
    data['object_cloud'] = object_cloud
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Updated data in {file_path}")

def visualize_and_create_video(pt_folder_path, pose_folder_path, ply_file_path, save_folder, ply_bowl_file_path, num_frames=500, frame_delay=0.1):
    """
    Visualizes point clouds from .pt files and applies object pose transformation using Open3D.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_control = vis.get_view_control()
    
    pt_files = [f"{i}.pt" for i in range(1, num_frames + 1)]
    poses , object_poses = load_pose_data(pose_folder_path, num_frames)
    print(poses)
    first_object_pcd = o3d.geometry.PointCloud()
    second_object_pcd = o3d.io.read_point_cloud(ply_file_path)
    second_object_pcd_copy = copy.deepcopy(second_object_pcd)
    print("second_object_pcd:", second_object_pcd)
    third_object_pcd = o3d.io.read_point_cloud(ply_bowl_file_path)
    third_object_pcd_copy = copy.deepcopy(third_object_pcd)
    combined_pcd = o3d.geometry.PointCloud()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(num_frames):
        pt_file_path = os.path.join(pt_folder_path, pt_files[i])
        print(f"Processing {pt_file_path} and pose file {i}...")
        sampled_pc_3d, tansform_pc = load_point_cloud(pt_file_path)
        print("sampled_pc_3d:", sampled_pc_3d)
        first_object_pcd.points = o3d.utility.Vector3dVector(sampled_pc_3d)
        
        pose_data = poses[i]
        print('i',[i])
        print('object_pose',pose_data)
        translation = pose_data["translation"]
        rotation = pose_data["rotation"]
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation 
        transformation_matrix[:3, 3] = translation  
        second_object_pcd.points = second_object_pcd_copy.points
        second_object_pcd.points = second_object_pcd.transform(transformation_matrix).points
        pose_data_bowl = np.array([[ 0.84176642 , 0.03681425 ,-0.53858545, -0.09505043],
                    [-0.5342848 , -0.08597439 ,-0.84092112, -0.69822697],
                    [-0.0772622 ,  0.9956163 , -0.05270113 , 0.13148672],
                    [ 0.    ,      0.    ,      0.     ,     1.        ]])

                    
        third_object_pcd.points = third_object_pcd_copy.points
        third_object_pcd.points = third_object_pcd.transform(pose_data_bowl).points
        
        second_points = np.asarray(second_object_pcd.points)
        third_points = np.asarray(third_object_pcd.points)
        combined_points = np.concatenate([second_points, third_points], axis=0)
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        if i == 1:
            vis.add_geometry(first_object_pcd)
            vis.add_geometry(combined_pcd)  

        vis.update_geometry(first_object_pcd)
        vis.update_geometry(combined_pcd)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(frame_delay)
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_folder_path', type=str, required=True) 
    parser.add_argument('--pose_folder_path', type=str, required=True)    
    parser.add_argument('--save_folder', type=str, required=True)  
    args = parser.parse_args()
    pt_folder_path = args.pt_folder_path
    pose_folder_path = args.pose_folder_path
    save_folder = args.save_folder
    ply_file_path = '/home/alan/project/c/c/object_cloud/test/cup_point_cloud_824.ply'
    ply_bowl_file_path = '/home/alan/project/c/c/object_cloud/kettle/kettle_point_cloud_200.ply'
    visualize_and_create_video(pt_folder_path, pose_folder_path, ply_file_path, save_folder, ply_bowl_file_path, num_frames=500, frame_delay=0.05)
