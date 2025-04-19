import open3d as o3d
import numpy as np
import os
import argparse

# Function to generate point cloud from the mesh
def generate_point_cloud(mesh_path, output_path, num_points=1024):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)
    
    center_of_mass = np.mean(points, axis=0)
    print(f"Center of the point cloud (pose): {center_of_mass}")
    colors = np.array([[1.0, 1.0, 0.0]] * points.shape[0])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    point_cloud_6d = o3d.geometry.PointCloud()
    point_cloud_6d.points = o3d.utility.Vector3dVector(points)
    

    # o3d.visualization.draw_geometries([point_cloud_6d])
    o3d.io.write_point_cloud(output_path, point_cloud_6d)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate point cloud from mesh.")
    parser.add_argument("--mesh_path", type=str, help="Path to the input mesh file.")
    parser.add_argument("--output_path", type=str, help="Path to save the output point cloud.")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample from the mesh (default: 1024).")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_point_cloud(args.mesh_path, args.output_path, args.num_points)
