import rospy
import sys
import os
sys.path.append('../policy/Diffusion-Policy')
from PIL import Image
from sensor_msgs.msg import JointState
import torch  
import numpy as np
import hydra
import yaml
from datetime import datetime
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner_leap_ur5 import DPRunner
from holodex.utils.network import ImageSubscriber
from termcolor import cprint
import dill
from scipy.spatial.transform import Rotation as R
import time

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

        self.color_image_subscribers, self.color_use_image_subscribers = [], []
        self.num_cams = 1
        self.episode_steps = 1000
        self.control_mode = "joint"

        for cam_num in range(self.num_cams):
            self.color_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/color_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_color_data_collector'.format(cam_num + 1)
                )
            )
            self.color_use_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/color_use_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_color_use_data_collector'.format(cam_num + 1)
                )
            )

        rospy.loginfo("Robot environment initialized.")

    def apply_dp(self, dp):
        cnt = 0
        start_time = time.time()
        start = time.time()
        observation = self.get_observation()
        dp.update_obs(observation)
        while cnt < self.episode_steps:
            # start = time.time()
            with torch.no_grad():
                action = dp.get_action()  
            
            print ("action.shape[0]",action.shape[0])
            # for time_step in range(action.shape[0]):  
            for time_step in range(6):
                current_action = action[time_step]  
                dt = time.time() - start
                time.sleep(max(1 / 10.0 - dt, 0))
                start = time.time()
                self.send_control_command(current_action) 
                observation = self.get_observation()
                dp.update_obs(observation)
            # cnt += action.shape[0]
            cnt += 6
        end_time = time.time()  
        total_time = end_time - start_time    
        print(f"Total time for completing {self.episode_steps} steps: {total_time:.2f} seconds.") 

    def get_observation(self):
        while self.hand.get_hand_position() is None or self.arm.get_arm_position() is None or self.color_use_image_subscribers[0].get_image() is None:
            time.sleep(0.1) 

        if self.control_mode == "joint":
            arm_joint_positions = np.array(self.arm.get_arm_position())
        elif self.control_mode == "tcp":
            arm_joint_positions = np.array(self.arm.get_tcp_position())
            rotation = R.from_euler('xyz', arm_joint_positions[3:])
            quaternion = rotation.as_quat()  # [x, y, z, w]
            arm_joint_positions = np.concatenate([arm_joint_positions[:3], quaternion])

        print ("arm_joint_positions",arm_joint_positions)
        hand_joint_positions = np.array(self.hand.get_hand_position())
        print ("hand_joint_positions",hand_joint_positions)
        joint_state = np.concatenate([hand_joint_positions, arm_joint_positions])

        image = self.color_use_image_subscribers[0].get_image()
        image = image[:, :, [2,1,0]]
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        print("joint_state shape:", joint_state.shape)
        print("image shape:", image.shape)
            
        return {'img': image, 'joint_state': joint_state}
    
    def quat_to_euler(self,input_array,order='xyz',to_degrees=False):
        quat = input_array[3:]
        rotation = R.from_quat(quat)
        euler_angles = rotation.as_euler(order)
        output_array = np.concatenate([input_array[:3],euler_angles])
        return output_array

    
    def send_control_command(self, action):
        if self.control_mode == "tcp":
            hand_position = action[:16] 
            arm_position = action[16:23]  
            arm_position = self.quat_to_euler(arm_position)
            rotation = R.from_euler('xyz', arm_position[3:])
            arm_position[3:] = rotation.as_rotvec()
            joint_position = self.arm.compute_joint(arm_position)
            cprint(f"tcp_position:{arm_position}", "red")
            cprint(f"arm_position:{joint_position}", "red")
            cprint(f"hand_position:{hand_position}", "red")
            self.hand.move(hand_position)
            self.arm.move(joint_position)
        elif self.control_mode == "joint":
            hand_position = action[:16] 
            arm_position = action[16:22]  
            cprint(f"arm_position:{arm_position}", "red")
            cprint(f"hand_position:{hand_position}", "red")
            self.hand.move(hand_position)
            self.arm.move(arm_position)
        # exit()
        rospy.loginfo(f"Control commands sent: Arm: {arm_position}, Hand: {hand_position}")
    
    def close(self):
        # 关闭 ROS 节点
        rospy.signal_shutdown("Task complete")
        print("Shutting down ROS node.")

#
def get_policy(checkpoint, output_dir, device):
    
    # load checkpoint
    payload = torch.load(open('./policy/Diffusion-Policy/'+checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model 
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

#
class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int):
        self.policy = get_policy(f'checkpoints/{task_name}/{checkpoint_num}.ckpt', None, 'cuda:0')
        # self.runner = DPRunner(output_dir=None)
        self.runner = DPRunner(n_obs_steps=4)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]


def test_policy(task_name, RobotEnv_class, dp: DP):

    env_instance = RobotEnv_class()  
    env_instance.apply_dp(dp) 

    env_instance.close()
    dp.runner.reset_obs()  


def main(usr_args):
    task_name = usr_args.task_name
    checkpoint_num = usr_args.checkpoint_num

    dp = DP(task_name, checkpoint_num, usr_args.data_num)

    test_policy(task_name, RobotEnv, dp)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('task_name', type=str, default='robot_task')
    parser.add_argument('checkpoint_num', type=int, default=1000)
    parser.add_argument('data_num', type=int, default=20)
    usr_args = parser.parse_args()
    
    main(usr_args)
