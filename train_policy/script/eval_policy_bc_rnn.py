import rospy
import sys
import os
sys.path.append('../policy/BCRNN')
from PIL import Image
from sensor_msgs.msg import JointState
import torch  
import numpy as np
import hydra
import yaml
from datetime import datetime
from argparse import ArgumentParser

from BCRNN.workspace.train_bc_rnn_image_workspace import TrainRobomimicImageWorkspace
from BCRNN.env_runner.bcrnn_runner import BCRNNRunner

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
        self.episode_steps = 1500
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

    def apply_bcrnn(self, bcrnn):
        cnt = 0
        start = time.time()
        observation = self.get_observation()
        bcrnn.update_obs(observation)
        while cnt < self.episode_steps:
            start = time.time()
            with torch.no_grad():
                action = bcrnn.get_action()  
            print("Frequency:", time.time() - start)
            for time_step in range(action.shape[0]):  
                current_action = action[time_step]  
                dt = time.time() - start
                time.sleep(max(1 / 10.0 - dt, 0))
                start = time.time()
                self.send_control_command(current_action) 
                observation = self.get_observation()
                bcrnn.update_obs(observation)
            cnt += action.shape[0]
    
    def get_observation(self):
        while self.hand.get_hand_position() is None or self.arm.get_arm_position() is None or self.color_use_image_subscribers[0].get_image() is None:
            time.sleep(0.1) 

        if self.control_mode == "joint":
            arm_joint_positions = np.array(self.arm.get_arm_position())

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
    
    def send_control_command(self, action):
        if self.control_mode == "joint":
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
    payload = torch.load(open('./policy/BCRNN/'+checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: TrainRobomimicImageWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

#
class BCRNN:
    def __init__(self, task_name, checkpoint_num: int, data_num: int):
        self.policy = get_policy(f'checkpoints/{task_name}/{checkpoint_num}.ckpt', None, 'cuda:0')
        self.runner = BCRNNRunner(n_obs_steps=12)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]


def test_policy(task_name, RobotEnv_class, bcrnn: BCRNN):

    env_instance = RobotEnv_class()  
    env_instance.apply_bcrnn(bcrnn) 

    env_instance.close()
    bcrnn.runner.reset_obs()  


def main(usr_args):
    task_name = usr_args.task_name
    checkpoint_num = usr_args.checkpoint_num

    bcrnn = BCRNN(task_name, checkpoint_num, usr_args.data_num)

    test_policy(task_name, RobotEnv, bcrnn)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('task_name', type=str, default='robot_task')
    parser.add_argument('checkpoint_num', type=int, default=1000)
    parser.add_argument('data_num', type=int, default=20)
    usr_args = parser.parse_args()
    
    main(usr_args)
