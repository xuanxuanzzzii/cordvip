import sys
sys.path.insert(0, './policy/CordViP/CordViP')
sys.path.append('./')

import torch  
import sapien.core as sapien
import os
import numpy as np
import hydra
import pathlib
import time
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import copy
from utils.hand_model import create_hand_model
import rospy
from sensor_msgs.msg import JointState
from holodex.utils.network import ImageSubscriber

from CordViP_policy import *
from termcolor import cprint
import pytorch3d.ops as torch3d_ops
import cv2
from geometry_msgs.msg import Pose

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
        rospy.Subscriber(OBJECT_POSE_TOPIC, Pose, self._callback_object_pose, queue_size = 1)
        # Hand controller initialization
        self.hand = Hand()
        self.arm = Arm(servo_mode=True,teleop=False,control_mode="joint",random_ur5_home=False)
        self.object_pose = None
        self.hand.home_robot()
        for i in range(5):
            self.hand.home_robot()
            time.sleep(0.1)
        time.sleep(5)

        self.num_cams = 1
        self.episode_steps = 1000

        self.base_path = "/home/c/Projects/DexTeleop/teleop/Real-Robo/demo/pc/reach_chicken/"
        folder_name = 1
        while os.path.exists(os.path.join(self.base_path, str(folder_name))):
            folder_name += 1
        self.pc_save_path = os.path.join(self.base_path, str(folder_name))

        self.frame_counter = 0 
        if not os.path.exists(self.pc_save_path):
            os.makedirs(self.pc_save_path) 
        self.fps = 30
        self.save_interval = 1 / self.fps
        if not hasattr(self, 'last_save_time'):
            self.last_save_time = time.time()

        self.hand_urdf = create_hand_model("leaphand", torch.device('cuda:0'), 1024)
        self.task = "reach_chicken"
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
        # self.view_control = self.vis.get_view_control()
        self.hand_pcd = o3d.geometry.PointCloud()
        self.combined_pcd = o3d.geometry.PointCloud()
        self.record_pcd = o3d.geometry.PointCloud()
        self.start = True
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


    def apply_CordViP(self, CordViP):
        cnt = 0
        start_time = time.time()
        observation = self.get_observation()
        CordViP.update_obs(observation)
        while cnt < self.episode_steps:
            # observation = self.get_observation()
            action = CordViP.get_action()  
            print("action:", action)
            print("action.shape:", action.shape)
            for time_step in range(action.shape[0]):  
                current_action = action[time_step]  
                # dt = time.time() - start
                # time.sleep(max(1 / 10.0 - dt, 0))
                # start = time.time()
                self.send_control_command(current_action) 
                observation = self.get_observation()
                CordViP.update_obs(observation)
                # self.view_control.set_lookat([-0.1, -0.7, 0.3])  # 设置摄像机的目标位置
                # self.view_control.set_up([0, 1, 0])      # 设置摄像机的上方向（通常为 [0, 1, 0]）
                # self.view_control.set_front([0, -0.5, 0.5])  # 设置摄像机的朝向
                self.vis.poll_events()
                self.vis.update_renderer()
        
            cnt += action.shape[0]
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time    
        print(f"Total time for completing {self.episode_steps} steps: {total_time:.2f} seconds.") 
     

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

        arm_joint_positions = np.array(self.arm.get_arm_position())
        hand_joint_positions = np.array(self.hand.get_hand_position())
        joint_state = np.concatenate([hand_joint_positions, arm_joint_positions])
        
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

        self.record_pc = np.concatenate([object_pc, hand_pc], axis=0)
        self.record_pcd.points = o3d.utility.Vector3dVector(self.record_pc)  
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval: 
            # file_name_pc = os.path.join(self.pc_save_path, f"record_pc_{self.frame_counter}.ply")
            # o3d.io.write_point_cloud(file_name_pc, self.record_pcd)
            self.frame_counter += 1

        return {'object_point_cloud': object_pc, "hand_point_cloud":hand_pc, 'joint_state': joint_state}
    
    def send_control_command(self, action):
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


def test_policy(RobotEnv_class, CordViP):
    global TASK
    
    env_instance = RobotEnv_class()
    env_instance.apply_CordViP(CordViP)

    env_instance.close()
    CordViP.env_runner.reset_obs()
  

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        '../policy/CordViP/CordViP/CordViP', 'config'))
)

def main(cfg):
    global TASK
    TASK = cfg.task.name
    print('Task name:', TASK)
    checkpoint_num = cfg.checkpoint_num
    
    CordViP = CordViP(cfg, checkpoint_num)
    test_policy(RobotEnv, CordViP)


if __name__ == "__main__":
    main()
