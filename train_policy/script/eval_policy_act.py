import rospy
import sys
import os
sys.path.append('../policy/act')
from PIL import Image
from sensor_msgs.msg import JointState
import torch  
import numpy as np
from holodex.utils.network import ImageSubscriber
from termcolor import cprint
import time
from constants import *
import torchvision.transforms as transforms
import pickle
from load_policy import get_policy

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
        rospy.init_node("eval_policy_lift3d_node", anonymous=True)
        print("RobotEnv")
        # Hand controller initialization
        print("hand inital")
        self.hand = Hand()
        print("arm inital")
        self.arm = Arm(servo_mode=True, teleop=False, control_mode="joint", random_ur5_home=False)
        for i in range(5):
            self.hand.home_robot()
            time.sleep(0.1)
        time.sleep(5)
      
        self.color_image_subscribers = []
        self.num_cams = 1
        self.episode_steps = 500
        self.control_mode = "joint"
        
        for cam_num in range(self.num_cams):
            self.color_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/color_use_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_color_data_collector'.format(cam_num + 1)
                )
            )

        rospy.loginfo("Robot environment initialized.")

    def apply_act(self, act_model):
        cnt = 0
        query_frequency = 30
        start = time.time()
        img, joint_state = self.get_observation()
        while cnt < self.episode_steps:
            start = time.time()
            with torch.no_grad():
                if cnt % query_frequency == 0:
                  
                    action = act_model.run_inference(img, joint_state)  
                else:
                    pass
            action = action.squeeze(0)
            for time_step in range(action.shape[0]):  
                print("action.shape:", action.shape)
                current_action = action[time_step]  
                current_action = current_action.cpu().numpy()
              
                dt = time.time() - start
                time.sleep(max(1 / 5.0 - dt, 0))
                start = time.time()
                self.send_control_command(current_action) 
                img, joint_state = self.get_observation()

            cnt += action.shape[0]
 
    def get_observation(self):
        while self.hand.get_hand_position() is None or self.arm.get_arm_position() is None or self.color_image_subscribers[0].get_image() is None:
            time.sleep(0.1) 

        if self.control_mode == "joint":
            arm_joint_positions = np.array(self.arm.get_arm_position())

        hand_joint_positions = np.array(self.hand.get_hand_position())
        joint_state = np.concatenate([hand_joint_positions, arm_joint_positions])
        
        image = self.color_image_subscribers[0].get_image()
        image = image[:, :, [2,1,0]]
        image_data = torch.from_numpy(image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)
        print ("image_data.shape",image_data.shape)
        
        ckpt_dir = "../policy/act/checkpoints/" # todo: task_name
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        pre_process = lambda joint_state: (joint_state - stats['qpos_mean']) / stats['qpos_std']
        joint_state = pre_process(joint_state)
        joint_state = torch.from_numpy(joint_state).float().cuda().unsqueeze(0)

        return image_data, joint_state

    def send_control_command(self, action):
        ckpt_dir = "../policy/act/checkpoints/" # todo: task_name
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        action = action
        action = post_process(action)

        hand_position = action[:16] 
        arm_position = action[16:22]  
        cprint(f"arm_position:{arm_position}", "red")
        cprint(f"hand_position:{hand_position}", "red")
        self.hand.move(hand_position)
        self.arm.move(arm_position)
        
        rospy.loginfo(f"Control commands sent: Arm: {arm_position}, Hand: {hand_position}")
    
    def close(self):
        # 关闭 ROS 节点
        rospy.signal_shutdown("Task complete")
        print("Shutting down ROS node.")


class ACT:
    def __init__(self):
        self.model = self.load_model()
        
    def load_model(self):
        model = get_policy()
        print("Model weights loaded successfully.")
        return model

    def run_inference(self, img, joint_state):
        print ("joint_state",joint_state.shape)
        print ("img",img.shape)
        output = self.model(joint_state,img)

        return output


def test_policy(RobotEnv_class, act_model:ACT):
    env_instance = RobotEnv_class()  
    
    env_instance.apply_act(act_model) 

    env_instance.close()

def main():
    act_model = ACT()
    test_policy(RobotEnv, act_model)


if __name__ == "__main__":
    main()
