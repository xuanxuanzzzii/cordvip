
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import trimesh
import logging
import argparse
import numpy as np
from estimater import *
from datareader import *
import argparse
import pyrealsense2 as rs 


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--mesh_file', type=str, default='/home/alan/project/c/c/chicken_mesh/texture2/combined_mesh.obj')
#   parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  bridge = CvBridge()
  rospy.init_node('pose_estimation_node', anonymous=True)
  color_image = None
  depth_image = None
  
  def color_callback(msg):
    global color_image
    color_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

  def depth_callback(msg):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough') 
    
  rospy.Subscriber("/camera/color/image_raw", Image, color_callback)
  rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)
  i = 0
  while True:
    mask = camera.get_mask(0).astype(bool)
    
    if i == 0:
        pose = est.register(K=camera.K, rgb=color_image, depth=depth_image, ob_mask=mask, iteration=args.est_refine_iter)
    else :
        pose = est.track_one(rgb=color_image, depth=depth_image, K=camera.K, iteration=args.track_refine_iter)  
        
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{[i]}.txt', pose.reshape(4,4))
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(camera.K, img=color_image, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color_image, ob_in_cam=center_pose, scale=0.1, K=camera.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)
        
    i += 1
    
#   for i in range(len(reader.color_files)):
#     logging.info(f'i:{i}')
#     color = reader.get_color(i)
#     depth = reader.get_depth(i)
#     if i==0:
#       mask = reader.get_mask(0).astype(bool)
#       pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

#       if debug>=3:
#         m = mesh.copy()
#         m.apply_transform(pose)
#         m.export(f'{debug_dir}/model_tf.obj')
#         xyz_map = depth2xyzmap(depth, reader.K)
#         valid = depth>=0.001
#         pcd = toOpen3dCloud(xyz_map[valid], color[valid])
#         o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
#     else:
#       pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

#     os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
#     np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

#     if debug>=1:
#       center_pose = pose@np.linalg.inv(to_origin)
#       vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
#       vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
#       cv2.imshow('1', vis[...,::-1])
#       cv2.waitKey(1)


#     if debug>=2:
#       os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
#       imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

