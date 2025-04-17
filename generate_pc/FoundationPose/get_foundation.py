from flask import Flask, request, jsonify
import threading
import cv2
import numpy as np

import rospy 
from cv_bridge import CvBridge

from estimater import *
from datareader import *
import argparse
import os
import trimesh
import time
import logging
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, Quaternion

from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

app = Flask(__name__)


class PoseEstimatorThread():
    def __init__(self, args):
        self.prev_pose = None
        self.object_pose = None
        self.running = True
        self.args = args

        set_logging_format()
        logging.disable(logging.CRITICAL)
        set_seed(0)

        self.mesh = trimesh.load(self.args.mesh_file)
        self.debug = self.args.debug
        self.debug_dir = self.args.debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)

        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx
        )
        
        self.data_folder = args.data_folder
        self.save_folder = args.save_folder
        
        self.file_range = range(0, 500)
        self.current_index = 0 
        self.files = [os.path.join(self.data_folder, f"{i}") for i in self.file_range]

        self.file_counter = 0 
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        
        logging.info("estimator initialization done")
        bridge = CvBridge() 
        self.K  = np.array([[597.21588135, 0., 324.15087891],
                    [ 0., 597.62249756, 237.27398682],
                    [ 0., 0., 1. ]])

        self.camera2base = np.load("./FoundationPose/expert_data/camera2base/20views_c2r.npy")
 
    def extract_pose_from_matrix(self, matrix):
        """get translation and quaternion"""
        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  #(x, y, z, w)
        return translation, quat

    def get_image(self):
            if self.current_index >= len(self.files):
                return None, None  
            file_path = self.files[self.current_index]

            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            color_image = data.get('camera_1_color_image')
            depth_image = data.get('camera_1_depth_image')

            color_image = np.asanyarray(color_image)
            depth_image = np.asanyarray(depth_image)
            
            depth_image = depth_image.astype(np.float32)
            depth_image = depth_image * 0.001  
            print('index',self.current_index)
            self.current_index += 1
            
            return color_image, depth_image  
        
    def save_pose_to_file(self, object2base):
        '''save pose as pkl file'''
        data = {'object_pose': object2base}
        file_name = f"{self.file_counter}"
        file_path = os.path.join(self.save_folder, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved pose to {file_path}")
        self.file_counter += 1


    def run(self):
        while self.running:
                color_image, depth_image = self.get_image()
                if  color_image is None or depth_image is None:
                    continue        
                if self.prev_pose is None:
                    open_path = f'{code_dir}/demo_data/flip_bottle/before_mask'
                    os.makedirs(open_path, exist_ok=True)  
                    cv2.imwrite(f'{open_path}/1.png', color_image)  

                    success = self.get_mask()  
                    if success:
                        print('Mask processing complete.')
                    else:
                        print('Error occurred during mask processing.')
                    mask_path = f'{code_dir}/demo_data/flip_bottle/mask/1.png'
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    cv2.imshow('Mask', mask)
                    if mask is None:
                        continue
                    pose = self.est.register(
                        K=self.K,
                        rgb=color_image,
                        depth=depth_image,
                        ob_mask=mask,
                        iteration=self.args.est_refine_iter
                    )
                    self.prev_pose = pose
                    object2camera = pose
                    object2base = np.dot(self.camera2base, object2camera)
                    self.prev_pose = pose
                    self.object_pose = object2base
                    self.save_pose_to_file(object2base)
                    print(f'save {self.file_counter} to {self.save_folder}!')
                else:
                    pose = self.est.track_one(
                        rgb=color_image,
                        depth=depth_image,
                        K=self.K,
                        iteration=self.args.track_refine_iter
                    )
                    object2camera = pose
                    object2base = np.dot(self.camera2base, object2camera)
                    self.prev_pose = pose
                    self.object_pose = object2base
                    self.save_pose_to_file(object2base)
                    print(f'save {self.file_counter} to {self.save_folder}!')

                if self.debug >= 1:
                    center_pose = pose @ np.linalg.inv(self.to_origin)
                    vis = draw_posed_3d_box(self.K, img=color_image, ob_in_cam=center_pose, bbox=self.bbox)
                    vis = draw_xyz_axis(
                        vis,
                        ob_in_cam=center_pose,
                        scale=0.1,
                        K=self.K,
                        thickness=3,
                        transparency=0,
                        is_input_rgb=True
                    )
                    cv2.imshow('Pose Estimation', vis)
                    key = cv2.waitKey(1)
                    if key & 0xFF == 27:  
                        break

    def stop(self):
        self.running = False
        
        
    def get_mask(self):
        '''Select mask by clicking on the screen'''
        sam = sam_model_registry["vit_h"](checkpoint=f'{code_dir}/demo_data/ckpt/sam_vit_h_4b8939.pth')
        predictor = SamPredictor(sam)
        image = Image.open(f'{code_dir}/demo_data/flip_bottle/before_mask/1.png')
        image_np = np.array(image)
        predictor.set_image(image_np)

        click_points = []
        click_labels = []

        mask_resized = np.zeros_like(image_np, dtype=np.uint8)
        def mouse_callback(event, x, y, flags, param):
            nonlocal click_points, click_labels, mask_resized
            if event == cv2.EVENT_LBUTTONDOWN: 
                click_points.append([x, y])
                click_labels.append(1)  
                print(f"Clicked point: ({x}, {y})")

                click_points_np = np.array(click_points)
                click_labels_np = np.array(click_labels)
                masks, _, _ = predictor.predict(click_points_np, click_labels_np)
                mask = masks[0]
                mask = mask.astype(np.uint8) * 255 

                mask_resized = np.array(Image.fromarray(mask).resize((image_np.shape[1], image_np.shape[0])))
                if len(mask_resized.shape) == 2:  
                    mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Generated Mask', mask_resized)

        cv2.imshow('Image', image_np)
        cv2.setMouseCallback('Image', mouse_callback)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
        print('close window')
        save_path = f'{code_dir}/demo_data/flip_bottle/mask'
        os.makedirs(save_path, exist_ok=True)  
        cv2.imwrite(f'{save_path}/1.png', mask_resized)  
        print("Mask processing complete.")
        return True  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default='./FoundationPose/demo_data/chicken_mesh/texture2/combined_mesh.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    args = parser.parse_args()
    pose_estimator = PoseEstimatorThread(args)
    pose_estimator.run() 
    app.run(host='127.0.0.1', port=5001)
   