import pickle
import open3d as o3d
import numpy as np
import time
import os 

def estimate_normals(point_cloud, k=30):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    
    o3d_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)  
    )
    
    normals = np.asarray(o3d_cloud.normals)
    return normals

def compute_aligned_distance(v_o, n_o, hand_points, gamma=1.0):
    distances = np.linalg.norm(hand_points - v_o, axis=1)  # (N,)
    
    directions = hand_points - v_o  # (N, 3)
    directions_normalized = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    alignment = np.abs(np.dot(directions_normalized, n_o))  # (N,)
    
    aligned_distances = np.exp(gamma * (1 - alignment)) * distances  # (N,)
    # aligned_distances = distances
    
    return np.min(aligned_distances)

def compute_contact_value(D, theta = 5):
    sigmoid = 1 / (1 + np.exp(-D * theta)) 
    return 1 - 2 * (sigmoid - 0.5)

def compute_contact_map(object_points, object_normals, hand_points, gamma=0.2):
    contact_map = []
    for v_o, n_o in zip(object_points, object_normals):
        D = compute_aligned_distance(v_o, n_o, hand_points, gamma)
        C = compute_contact_value(D, theta=10)
        contact_map.append(C)
    return np.array(contact_map)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def process_data(load_dir, num, current_ep, num_files):
    while os.path.isdir(load_dir + f'/demonstration_{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        while os.path.exists(load_dir + f'/demonstration_{current_ep}' + f'/{file_num}') and file_num < num_files:
            file_path = load_dir + f'/demonstration_{current_ep}' + f'/{file_num}'
            
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            
            object_point_cloud = data.get('object_cloud', None)
            hand_point_cloud = data.get('hand_cloud', None)

            if object_point_cloud is not None and hand_point_cloud is not None:
                object_normals = estimate_normals(object_point_cloud)
                contact_map = compute_contact_map(object_point_cloud, object_normals, hand_point_cloud, gamma=1)
                data['contact_map'] = contact_map
                save_pickle(data, file_path)
            
            file_num += 1  
        current_ep += 1 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process point cloud data and compute contact maps.")
    parser.add_argument('load_dir', type=str, help="Path to the directory containing the recorded data.")
    parser.add_argument('num', type=int, help="Number of files per episodes to process.")
    parser.add_argument('current_ep', type=int, help="The start number of files to process.")
    parser.add_argument('num_files', type=int, help="Number of episodes to process.")

    args = parser.parse_args()

    process_data(args.load_dir, args.num, args.num_files)
