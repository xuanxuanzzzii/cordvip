from omni.isaac.kit import SimulationApp
import argparse
import sys
import carb
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path
import omni.usd as usd_utils
from pxr import UsdPhysics, Sdf, Gf
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation.articulation_motion_policy import ArticulationMotionPolicy
from omni.isaac.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
)
from omni.isaac.core.prims import XFormPrim
from typing import List, Optional, Sequence, Union
from termcolor import cprint
# from omni.isaac.universal_robots import KinematicsSolver

import os
from typing import Optional

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver
from omni.isaac.motion_generation.articulation_motion_policy import ArticulationAction

class UR5KinematicsSolver(ArticulationKinematicsSolver):
    """Kinematics Solver for UR10 robot.  This class loads a LulaKinematicsSovler object

    Args:
        robot_articulation (Articulation): An initialized Articulation object representing this UR10
        end_effector_frame_name (Optional[str]): The name of the UR10 end effector.  If None, an end effector link will
            be automatically selected.  Defaults to None.
        attach_gripper (Optional[bool]): If True, a URDF will be loaded that includes a suction gripper.  Defaults to False.
    """

    def __init__(
        self,
        robot_articulation: Articulation,
        end_effector_frame_name: Optional[str] = None,
        attach_gripper: Optional[bool] = False,
    ) -> None:

        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")

        robot_urdf_path = "assets/ur5_robot.urdf"
        robot_description_yaml_path = os.path.join(
            mg_extension_path, "motion_policy_configs/universal_robots/ur5/rmpflow/ur5_robot_description.yaml"
        )

        self._kinematics = LulaKinematicsSolver(
            robot_description_path=robot_description_yaml_path, urdf_path=robot_urdf_path
        )

        if end_effector_frame_name is None:
            if attach_gripper:
                end_effector_frame_name = "ee_suction_link"
            else:
                end_effector_frame_name = "ee_link"

        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)

        return

class Ur5Leaphand(Robot):
    def __init__(self, world:World, translation:np.ndarray, orientation:np.ndarray):
        # define world
        self.world = world
        # define DexLeft name
        self._name = "Ur5Leaphand"
        # define DexLeft prim
        self._prim_path = "/World/ur5_leaphand"
        self.hand_link_name = "hand_link"  
        self.hand_prim_path = f"{self._prim_path}/{self.hand_link_name}"
        self._prim_path = "/World/ur5_leaphand"
        self.ee_link_name = "ee_link"  
        self.ee_prim_path = f"{self._prim_path}/{self.ee_link_name}"

        # get DexLeft usd file path
        self.asset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/ur5_leaphand.usd")
        # define DexLeft positon
        self.translation = translation
        # define DexLeft orientation
        self.orientation = orientation
        
        # add DexLeft USD to stage
        add_reference_to_stage(self.asset_file, self._prim_path)
        # initialize DexLeft Robot according to USD file loaded in stage
        super().__init__(
            prim_path=self._prim_path,
            name=self._name,
            translation=self.translation,
            orientation=self.orientation,
            articulation_controller = None
        )
        # add DexLeft to the scene
        self.world.scene.add(self)
        
        # # inverse kinematics control
        self.ki_solver = UR5KinematicsSolver(self, end_effector_frame_name="ee_link")
        self.end_effector = XFormPrim(self.ee_prim_path, "end_effector")

        # RMPFlow control
        self.rmp_config = load_supported_motion_policy_config("UR5", "RMPflow")
        self.rmpflow = RmpFlow(**self.rmp_config)
        self.rmpflow.set_robot_base_pose(self.translation, self.orientation)
        self.articulation_rmpflow = ArticulationMotionPolicy(self, self.rmpflow, default_physics_dt = 1 / 60.0)
        self._articulation_controller=self.get_articulation_controller()
        
    def initialize(self, physics_sim_view):
        # initialize robot
        super().initialize(physics_sim_view)
        # reset default status
        self.set_default_state(position=self.translation, orientation=self.orientation)
        self.set_joints_default_state(
            positions = np.array([
                -1.57, -1.57, -1.57, -1.57, 1.57, 0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                ])
        )
        # get arm_dof names and arm_dof indices
        self.arm_dof_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]
        arm_default_positions = np.array([-1.57, -1.57, 1.57, -1.57, 1.57, 0.0])
        self.arm_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.arm_dof_names]
        
        # get hand_dof names and hand_dof indices
        self.hand_dof_names = [
            "joint1", "joint0", "joint2", "joint3",
            "joint5", "joint4", "joint6", "joint7", 
            "joint9", "joint8", "joint10", "joint11",
            "joint12", "joint13", "joint14", "joint15"
        ]
        hand_default_positions = np.array([0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0,])
        self.hand_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.hand_dof_names]
       
    def get_hand_positions(self):
        return self.get_joint_positions(self.hand_dof_indices)
    
    def get_arm_positions(self):
        return self.get_joint_positions(self.arm_dof_indices)

    def get_tcp_positions(self):
        '''
        get current end_effector_position and end_effector orientation
        '''
        position, orientation = self.end_effector.get_local_pose()
        print("origin orientation:", orientation)
        orientation = np.roll(np.array(orientation), 3)
        rotation = R.from_quat(orientation) 
        euler_angles = rotation.as_euler("xyz")
        euler_angles = euler_angles
        print("euler_angles", euler_angles)
        arm_ee_pos = np.concatenate((position, euler_angles))
        return arm_ee_pos

    def move(self, target_positions: np.ndarray, joint_indices: Optional[Union[List, np.ndarray]] = None):
        target_positions = np.copy(target_positions)
        cprint(target_positions, "green")
        # self.set_joint_positions(target_positions, joint_indices)
        self._articulation_controller.apply_action(ArticulationAction(joint_positions=target_positions,
                                                                      joint_velocities=None,
                                                                      joint_efforts=None,
                                                                      joint_indices=joint_indices))
        cprint(self.get_joint_positions(self.arm_dof_indices), "red")
    
    def compute_joint(self, tcp_position):
        '''
        get current joint positions
        '''
        euler_angles = tcp_position[3:6]
        rotation = R.from_euler('xyz', euler_angles)  
        quaternion = rotation.as_quat() 
        quaternion = quaternion[[3,0,1,2]]
        print("quaternion", quaternion)
        desired_joint_angles,_ = self.ki_solver.compute_inverse_kinematics(tcp_position[:3],target_orientation=quaternion,position_tolerance=0.01)
        desired_joint_angles_positions = desired_joint_angles.joint_positions 
        return desired_joint_angles_positions
