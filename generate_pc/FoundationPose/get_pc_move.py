import open3d as o3d
import numpy as np
import pickle
import os

pose_folder = './expert_data/pick_chicken/demonstration_1'
vis = o3d.visualization.Visualizer()
vis.create_window()

object_pcd = o3d.geometry.PointCloud()
hand_pcd = o3d.geometry.PointCloud()

for i in range(0, 499):
    pkl_file = os.path.join(pose_folder, f'{i}')
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            object_cloud = data.get('object_cloud')  
            hand_cloud = data.get('hand_cloud')  
            if isinstance(object_cloud, np.ndarray) and isinstance(hand_cloud, np.ndarray):
                object_pcd.points = o3d.utility.Vector3dVector(object_cloud)
                hand_pcd.points = o3d.utility.Vector3dVector(hand_cloud)
                if i == 0:
                    vis.add_geometry(object_pcd)  
                    vis.add_geometry(hand_pcd) 
                vis.update_geometry(object_pcd)
                vis.update_geometry(hand_pcd)

                vis.poll_events()
                vis.update_renderer()

vis.run()
vis.destroy_window()
