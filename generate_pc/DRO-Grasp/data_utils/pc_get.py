import open3d as o3d
import numpy as np

import os

output_dir = "/home/alan/project/c/c/object_cloud/test/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


mesh = o3d.io.read_triangle_mesh("./FoundationPose/object_mesh/1/mesh_with_texture.obj")
pcd = mesh.sample_points_uniformly(number_of_points=824)

points = np.asarray(pcd.points)
center_of_mass = np.mean(points, axis=0)

print(f"Center of the point cloud (pose): {center_of_mass}")
colors = np.array([[1.0, 1.0, 0.0]] * points.shape[0])  
points_rgb = np.hstack((points, colors)) 
pcd.colors = o3d.utility.Vector3dVector(colors) 
point_cloud_6d = o3d.geometry.PointCloud()
point_cloud_6d.points = o3d.utility.Vector3dVector(points)  

o3d.visualization.draw_geometries([point_cloud_6d])
o3d.io.write_point_cloud("./object_cloud/test/cup_point_cloud_824.ply", point_cloud_6d)