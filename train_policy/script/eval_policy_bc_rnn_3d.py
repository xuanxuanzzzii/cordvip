import rospy
import sys
import os
sys.path.append('../policy/BCRNN_3D')
from PIL import Image
from sensor_msgs.msg import JointState
import torch  
import numpy as np
import hydra
import yaml
from datetime import datetime
from argparse import ArgumentParser

from BCRNN_3D.workspace.train_bc_rnn_pc_workspace import TrainRobomimicPointCloudWorkspace
from BCRNN_3D.env_runner.bcrnn3d_runner import BCRNN3DRunner

from holodex.utils.network import ImageSubscriber
from termcolor import cprint
import dill
from scipy.spatial.transform import Rotation as R
import time
import pytorch3d.ops as torch3d_ops
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
        self.arm = Arm(servo_mode=True, teleop=False, control_mode="joint", random_ur5_home=False)
        for i in range(5):
            self.hand.home_robot()
            time.sleep(0.1)
        time.sleep(5)

        self.color_image_subscribers, self.depth_image_subscribers = [], []
        self.num_cams = 1
        # self.episode_steps = 1500
        self.episode_steps = 3000
        self.control_mode = "joint"

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

    def apply_bcrnn3d(self, bcrnn3d):
        cnt = 0
        start = time.time()
        observation = self.get_observation()
        bcrnn3d.update_obs(observation)
        while cnt < self.episode_steps:
            start = time.time()
            with torch.no_grad():
                action = bcrnn3d.get_action()  
            print("Frequency:", time.time() - start)
            for time_step in range(action.shape[0]):  
                current_action = action[time_step]  
                dt = time.time() - start
                time.sleep(max(1 / 5.0 - dt, 0))
                start = time.time()
                self.send_control_command(current_action) 
                observation = self.get_observation()
                bcrnn3d.update_obs(observation)
            cnt += action.shape[0]
    
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
        if self.control_mode == "joint":
            arm_joint_positions = np.array(self.arm.get_arm_position())

        print ("arm_joint_positions",arm_joint_positions)
        hand_joint_positions = np.array(self.hand.get_hand_position())
        print ("hand_joint_positions",hand_joint_positions)
        joint_state = np.concatenate([hand_joint_positions, arm_joint_positions])

        image = self.color_image_subscribers[0].get_image()
        image = image[:, :, [2,1,0]]
        
        depth = self.depth_image_subscribers[0].get_image()
        if image is not None and depth is not None:
            point_cloud = self.generate_point_cloud(image, depth)
            point_cloud = self.preprocess_point_cloud(point_cloud)
            
        return {'point_cloud': point_cloud[:,:3], 'joint_state': joint_state}
    
    def send_control_command(self, action):
        if self.control_mode == "joint":
            hand_position = action[:16] 
            arm_position = action[16:22]  
            
            cprint(f"arm_position:{arm_position}", "red")
            cprint(f"hand_position:{hand_position}", "red")
            for i in range(4):
                self.hand.move(hand_position)
                self.arm.move(arm_position)
                time.sleep(0.02)
        # exit()
        rospy.loginfo(f"Control commands sent: Arm: {arm_position}, Hand: {hand_position}")
    
    def close(self):
        rospy.signal_shutdown("Task complete")
        print("Shutting down ROS node.")

#
def get_policy(checkpoint, output_dir, device):
    
    # load checkpoint
    payload = torch.load(open('./policy/BCRNN_3D/'+checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: TrainRobomimicPointCloudWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

#
class BCRNN3D:
    def __init__(self, task_name, checkpoint_num: int, data_num: int):
        self.policy = get_policy(f'checkpoints/{task_name}/{checkpoint_num}.ckpt', None, 'cuda:0')
        self.runner = BCRNN3DRunner(n_obs_steps=10)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def test_policy(task_name, RobotEnv_class, bcrnn3d: BCRNN3D):

    env_instance = RobotEnv_class()  
    env_instance.apply_bcrnn3d(bcrnn3d) 

    env_instance.close()
    bcrnn3d.runner.reset_obs()  


def main(usr_args):
    task_name = usr_args.task_name
    checkpoint_num = usr_args.checkpoint_num

    bcrnn3d = BCRNN3D(task_name, checkpoint_num, usr_args.data_num)

    test_policy(task_name, RobotEnv, bcrnn3d)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('task_name', type=str, default='robot_task')
    parser.add_argument('checkpoint_num', type=int, default=1000)
    parser.add_argument('data_num', type=int, default=20)
    usr_args = parser.parse_args()
    
    main(usr_args)
