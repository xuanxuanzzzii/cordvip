import rospy
import sys
import os
sys.path.append('../policy/act_3d_ours')
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
import pytorch3d.ops as torch3d_ops
import open3d as o3d

from utils.hand_model import create_hand_model
import copy
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
import argparse

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
    def __init__(self,ckpt_dir="../policy/act_3d_ours/checkpoints/"):
        rospy.init_node("eval_policy_lift3d_node", anonymous=True)
        rospy.Subscriber(OBJECT_POSE_TOPIC, Pose, self._callback_object_pose, queue_size = 1)
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
        self.ckpt_dir = ckpt_dir
        self.num_cams = 1
        self.episode_steps = 750
        self.object_pose = None
        self.hand_urdf = create_hand_model("leaphand", torch.device('cuda:0'), 1024)
        self.task = "Assembly"
        if self.task == "reach_chicken":
            ply_file_path = './assets/object_pc/chicken_4_point_cloud_824.ply'
            ply_bowl_file_path = './assets/object_pc/bowl_flatten_point_cloud_200.ply'
            self.object_pc1 = o3d.io.read_point_cloud(ply_file_path)
            self.object_pc1_copy = copy.deepcopy(self.object_pc1)
            self.object_pc2 = o3d.io.read_point_cloud(ply_bowl_file_path)
            self.object_pc2_copy = copy.deepcopy(self.object_pc2)
        elif self.task == "flip_cup":
            ply_file_path = './assets/cup_pc/cup_point_cloud_1024.ply'
            self.object_pc = o3d.io.read_point_cloud(ply_file_path)
            self.object_pc_copy = copy.deepcopy(self.object_pc)
        elif self.task == "Assembly":
            ply_file_path = './assets/kettle_pc/cup_point_cloud_824.ply'
            ply_kettle_file_path = './assets/kettle_pc/kettle_point_cloud_200.ply'
            self.object_pc1 = o3d.io.read_point_cloud(ply_file_path)
            self.object_pc1_copy = copy.deepcopy(self.object_pc1)
            self.object_pc2 = o3d.io.read_point_cloud(ply_kettle_file_path)
            self.object_pc2_copy = copy.deepcopy(self.object_pc2)
        elif self.task == "Articulated_manip":
            ply_file_path = './assets/object_cloud/box/box_point_cloud_1024.ply'
            self.object_pc = o3d.io.read_point_cloud(ply_file_path)
            self.object_pc_copy = copy.deepcopy(self.object_pc)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.hand_pcd = o3d.geometry.PointCloud()
        self.combined_pcd = o3d.geometry.PointCloud()
        self.start = True

        self.control_mode = "joint"

        rospy.loginfo("Robot environment initialized.")
    
    def _callback_object_pose(self, msg):
        translation = [msg.position.x, msg.position.y, msg.position.z]
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        rotation = R.from_quat(quat)  
        rotation_matrix = rotation.as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 3] = translation
        transform_matrix[0:3, 0:3] = rotation_matrix
        self.object_pose = transform_matrix

    def apply_act(self, act_model):
        cnt = 0
        query_frequency = 30
        start = time.time()
        hand_pc, object_pc, joint_state = self.get_observation()
        # print ("point_cloud:",point_cloud)
        while cnt < self.episode_steps:
            start = time.time()
            with torch.no_grad():
                if cnt % query_frequency == 0:
                    action = act_model.run_inference(hand_pc, object_pc, joint_state)  
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
                hand_pc, object_pc, joint_state = self.get_observation()
                self.vis.poll_events()
                self.vis.update_renderer()

            cnt += action.shape[0]
    
    def get_hand_point_cloud(self, q, num_points):
        q = torch.tensor(q, dtype=torch.float32, device="cuda:0")
        q = q[[0,1,2,3,4,5,7,6,8,9,11,10,12,13,15,14,16,17,18,19,20,21]]
        print(1)
        sampled_pc, sampled_pc_index, sampled_transform = self.hand_urdf.get_sampled_pc(q=q, num_points=num_points)
        print(2)
        return sampled_pc.cpu().numpy()

    def get_observation(self):
        while self.hand.get_hand_position() is None or self.arm.get_arm_position() is None or self.object_pose is None:
            time.sleep(0.1) 
        
        # joint_state
        if self.control_mode == "joint":
            arm_joint_positions = np.array(self.arm.get_arm_position())

        hand_joint_positions = np.array(self.hand.get_hand_position())
        joint_state = np.concatenate([hand_joint_positions, arm_joint_positions])

        ckpt_dir = self.ckpt_dir
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        pre_process = lambda joint_state: (joint_state - stats['qpos_mean']) / stats['qpos_std']
        joint_state = pre_process(joint_state)
        joint_state = torch.from_numpy(joint_state).float().cuda().unsqueeze(0)
        
        # point_cloud
        object_pose = np.array(self.object_pose)
        if self.task == "reach_chicken":
            self.object_pc1.points = self.object_pc1_copy.points
            self.object_pc1.points = self.object_pc1.transform(object_pose).points

            pose_data_bowl = np.array([ [ 0.01003577,  0.02600824,  0.99961119, -0.29278216],
                                    [ 0.9996347,  0.02483542, -0.01068148, -0.58358154 ],
                                    [-0.02510479,  0.99935342,  -0.02575032,  0.13154158],
                                    [ 0.  ,        0. ,         0. ,         1.        ]])
            self.object_pc2.points = self.object_pc2_copy.points
            self.object_pc2.points = self.object_pc2.transform(pose_data_bowl).points
            object_pointcloud1 = np.asarray(self.object_pc1.points)
            object_pointcloud2 = np.asarray(self.object_pc2.points)

            object_pc = np.concatenate([object_pointcloud1, object_pointcloud2], axis=0)
        elif self.task == "flip_cup":
            self.object_pc.points = self.object_pc_copy.points
            self.object_pc.points = self.object_pc.transform(object_pose).points
            object_pc = np.asarray(self.object_pc.points)
        elif self.task == "Assembly":
            self.object_pc1.points = self.object_pc1_copy.points
            self.object_pc1.points = self.object_pc1.transform(object_pose).points

            pose_data_bowl = np.array([[ 0.84176642 , 0.03681425 ,-0.53858545, -0.09505043],
                    [-0.5342848 , -0.08597439 ,-0.84092112, -0.69822697],
                    [-0.0772622 ,  0.9956163 , -0.05270113 , 0.13148672],
                    [ 0.    ,      0.    ,      0.     ,     1.        ]])
            self.object_pc2.points = self.object_pc2_copy.points
            self.object_pc2.points = self.object_pc2.transform(pose_data_bowl).points
            object_pointcloud1 = np.asarray(self.object_pc1.points)
            object_pointcloud2 = np.asarray(self.object_pc2.points)

            object_pc = np.concatenate([object_pointcloud1, object_pointcloud2], axis=0)
        elif self.task == "Articulated_manip":
            self.object_pc.points = self.object_pc_copy.points
            self.object_pc.points = self.object_pc.transform(object_pose).points
            object_pc = np.asarray(self.object_pc.points)

        self.combined_pcd.points = o3d.utility.Vector3dVector(object_pc)

        joint_state_tmp = np.concatenate([arm_joint_positions, hand_joint_positions])
        hand_pc = self.get_hand_point_cloud(q=joint_state_tmp, num_points=1024)[:, :3]
        self.hand_pcd.points = o3d.utility.Vector3dVector(hand_pc)
        print("hand_pc:", hand_pc.shape)
        print("object_pc:", object_pc.shape)

        if self.start == True:
            self.vis.add_geometry(self.combined_pcd)
            self.vis.add_geometry(self.hand_pcd)
            self.start = False
        self.vis.update_geometry(self.hand_pcd)
        self.vis.update_geometry(self.combined_pcd)
        hand_pc = torch.from_numpy(hand_pc).float().unsqueeze(0)
        object_pc = torch.from_numpy(object_pc).float().unsqueeze(0)
        return hand_pc, object_pc, joint_state

    def send_control_command(self, action):
        ckpt_dir = self.ckpt_dir
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
        rospy.signal_shutdown("Task complete")
        print("Shutting down ROS node.")


class ACT:
    def __init__(self,ckpt_dir="../policy/act_3d_ours/checkpoints/"):
        self.model = self.load_model()
        self.ckpt_dir = ckpt_dir
        
    def load_model(self):
        model = get_policy(self.ckpt_dir)
        print("Model weights loaded successfully.")
        return model

    def run_inference(self, hand_pc, object_pc, joint_state):
        print ("joint_state",joint_state.shape)
        print ("hand_pc",hand_pc.shape)
        print ("object_pc",object_pc.shape)
        device = torch.device("cuda")
        output = self.model(joint_state.to(device),hand_pc.to(device),object_pc.to(device))

        return output

def test_policy(RobotEnv_class, act_model: ACT, ckpt_dir=None):
    env_instance = RobotEnv_class(ckpt_dir=ckpt_dir) 
    env_instance.apply_act(act_model)
    env_instance.close()

def main(ckpt_dir=None):
    act_model = ACT(ckpt_dir=ckpt_dir)
    test_policy(RobotEnv, act_model, ckpt_dir=ckpt_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../policy/act_3d_ours/checkpoints/",
                       help='Directory containing checkpoint and stats files')
    args = parser.parse_args()

    main(ckpt_dir=args.dir)