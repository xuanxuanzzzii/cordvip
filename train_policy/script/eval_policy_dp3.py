import sys
sys.path.insert(0, './policy/3D-Diffusion-Policy/3D-Diffusion-Policy')
sys.path.append('./')

import torch  
import sapien.core as sapien
import os
import numpy as np
import hydra
import pathlib
import time
from train import TrainDP3Workspace
import rospy
from sensor_msgs.msg import JointState
from holodex.utils.network import ImageSubscriber

from dp3_policy import *
from termcolor import cprint
import pytorch3d.ops as torch3d_ops
import cv2
TASK = None

from constants import *
if HAND_TYPE is not None:
    hand_module = __import__("holodex.robot.hand")
    Hand_module_name = f'{HAND_TYPE}Hand'
    Hand = getattr(hand_module.robot, Hand_module_name)
    hand_type = HAND_TYPE.lower()

if ARM_TYPE is not None:
    arm_module = __import__("holodex.robot.arm")
    Arm_module_name = f'{ARM_TYPE}Arm'
    Arm = getattr(arm_module.robot, Arm_module_name)

class RobotEnv:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)

        # Hand controller initialization
        self.hand = Hand()
        self.arm = Arm(servo_mode=True,teleop=False,control_mode="joint",random_ur5_home=False)
        self.hand.home_robot()
        for i in range(5):
            self.hand.home_robot()
            time.sleep(0.1)
        time.sleep(5)

        self.color_image_subscribers, self.depth_image_subscribers = [], []
        self.num_cams = 1
        self.episode_steps = 1000

        for cam_num in range(self.num_cams):
            self.color_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/color_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_color_data_collector'.format(cam_num + 1)
                )
            )
            self.depth_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/depth_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_depth_data_collector'.format(cam_num + 1),
                    color = False
                )
            )

        rospy.loginfo("Robot environment initialized.")

    def apply_dp3(self, dp3):
        cnt = 0
        start_time = time.time()
        # start = time.time()
        observation = self.get_observation()
        dp3.update_obs(observation)
        while cnt < self.episode_steps:
            action = dp3.get_action()  
            for time_step in range(6):
                current_action = action[time_step]  
                self.send_control_command(current_action) 
                observation = self.get_observation()
                dp3.update_obs(observation)
            # cnt += action.shape[0]
            cnt += 6
        end_time = time.time()  
        total_time = end_time - start_time    
        print(f"Total time for completing {self.episode_steps} steps: {total_time:.2f} seconds.") 
    
    def generate_point_cloud(self, color_image, depth_image):
        rgb = color_image
        depth = depth_image

        depth_scale = 0.001
        depth_image = depth.astype(np.float32) * depth_scale
        
        depth_threshold = 1.2  
        depth_image[depth_image > depth_threshold] = 0  

        fx = 597.216
        fy = 597.622
        ppx = 324.151
        ppy = 237.274
        
        height, width = depth_image.shape
        
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_image
        x3 = (x - ppx) * z / fx
        y3 = (y - ppy) * z / fy
        
        points_old = np.vstack((x3.flatten(), y3.flatten(), z.flatten())).T
        points = np.ones((points_old.shape[0], 4))
        points[:,:3] = points_old
        
        matrix = np.array([[-0.52939579, 0.41285064,  -0.74114401, 0.5394448 ],
                                [0.84833125,  0.26647988, -0.45751783,  -0.23682375],
                                [0.00861344, -0.87094364, -0.49130743,  0.55116675],
                                [ 0.,          0.,          0.,          1.]])
        
        points = np.transpose(points)
        
        points = np.transpose(matrix @ points)
        
        points = points[:, :3]
        colors = rgb.reshape(-1, 3) / 255.0


        # x_now = points[:, 0]
        # y_now = points[:, 1]
        # z_now = points[:, 2]
        # mask = (x_now.flatten() > - 0.15 ) & (z_now.flatten() > 0.10) & (y_now.flatten() > - 0.4 ) & (y_now.flatten() < 0.3) 
        # x_mask = x_now.flatten() > 0
        # in_y_range = y_now.flatten() > 0.3
        # in_z_range = z_now.flatten() < 0.05
        # black_color_mask = np.linalg.norm(colors, axis=1) < 0.2
        # threshold = 0.35
        # black_color_mask = (colors[:, 0] < threshold) & (colors[:, 1] < threshold) & (colors[:, 2] < threshold)
        # y_mask = ~(in_y_range & black_color_mask)
        # z_mask = ~(in_z_range & black_color_mask)
        
        # final_mask = mask & y_mask & z_mask
        
        # point_cloud.points = o3d.utility.Vector3dVector(points[final_mask])
        # point_cloud.colors = o3d.utility.Vector3dVector(colors[final_mask])

        # num_points = np.asarray(point_cloud.points).shape[0]
        # print(num_points)

        point_clouds = np.concatenate((points, colors * 255), axis=-1)
        
        return point_clouds

    def farthest_point_sampling(self, points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices

    def preprocess_point_cloud(self, points, use_cuda=True):
        
        num_points = 1024

        WORK_SPACE = [
            [-0.40, 0.1],
            [-0.7, -0.4],
            [0.10, 0.51]
        ]
        
        # crop
        points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                    (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                    (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

        
        points_xyz = points[..., :3]
    
        points_xyz, sample_indices = self.farthest_point_sampling(points_xyz, num_points, use_cuda)
        sample_indices = sample_indices.cpu()
        points_rgb = points[sample_indices, 3:][0]
        points = np.hstack((points_xyz, points_rgb))

        return points


    def get_observation(self):
        while self.hand.get_hand_position() is None or self.arm.get_arm_position() is None or self.color_image_subscribers[0].get_image() is None or self.depth_image_subscribers[0].get_image() is None:
            time.sleep(0.1) 

        arm_joint_positions = np.array(self.arm.get_arm_position())
        hand_joint_positions = np.array(self.hand.get_hand_position())
        joint_state = np.concatenate([hand_joint_positions, arm_joint_positions])

        image = self.color_image_subscribers[0].get_image()
        image = image[:, :, [2,1,0]]
        
        depth = self.depth_image_subscribers[0].get_image()
        if image is not None and depth is not None:
            point_cloud = self.generate_point_cloud(image, depth)
            point_cloud = self.preprocess_point_cloud(point_cloud)
        return {'point_cloud': point_cloud[:, :3], 'joint_state': joint_state}
    
    def send_control_command(self, action):
        hand_position = action[:16] 
        arm_position = action[16:22]  
        
        cprint(f"arm_position:{arm_position}", "red")
        cprint(f"hand_position:{hand_position}", "red")
        self.hand.move(hand_position)
        self.arm.move(arm_position)

        rospy.loginfo(f"Control commands sent: Arm: {arm_position}, Hand: {hand_position}")
    
    def close(self):
        rospy.signal_shutdown("Task complete")
        print("Shutting down ROS node.")


def test_policy(RobotEnv_class, dp3):
    global TASK
    
    env_instance = RobotEnv_class()
    env_instance.apply_dp3(dp3)

    env_instance.close()
    dp3.env_runner.reset_obs()
  

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        '../policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d', 'config'))
)

def main(cfg):
    global TASK
    TASK = cfg.task.name
    print('Task name:', TASK)
    checkpoint_num = cfg.checkpoint_num
    
    dp3 = DP3(cfg, checkpoint_num)
    test_policy(RobotEnv, dp3)


if __name__ == "__main__":
    main()
