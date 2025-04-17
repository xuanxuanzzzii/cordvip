import os.path as path
import holodex
import numpy as np
from math import pi as PI
import torch

# Retarget type
RETARGET_TYPE = "dexpilot"

# dexpilot
SMOOTH_FACTOR = 0.4 # 0.99 

# Robot type
HAND_TYPE = "Leap"
# HAND_TYPE = None
# ARM_TYPE = "Jaka"
ARM_TYPE = "UR5"
# ARM_TYPE = None

UR5_IP = "192.168.0.22"

# Ur5
UR5_JOINT_STATE_TOPIC = '/ur5/joint_states'
UR5_COMMANDED_JOINT_STATE_TOPIC = '/ur5/commanded_joint_states'
UR5_EE_POSE_TOPIC = '/ur5/ee_pose'
UR5_DOF = 6
UR5_POSITIONS = {
    'home':[-PI/2, -PI/2, -PI/2, -PI/2, PI/2, 0],
    'tcp_home': [-0.108, 0.485, 0.588, 1.189, 1.248, -1.229] #reorient
}

# Leap hand
LEAP_JOINT_STATE_TOPIC = '/leaphand_node/joint_states'
LEAP_COMMANDED_JOINT_STATE_TOPIC = '/leaphand_node/hand_command_joint_states'
LEAP_JOINTS_PER_FINGER = 4
LEAP_JOINT_OFFSETS = {
    'index': 0,
    'middle': 4,
    'ring': 8,
    'thumb': 12
}
# 16 degree 0 position
# LEAP_HOME_POSITION = [ 0.07516766,  0.05522585, -0.01380324, -0.16873527,  0.13652682,
#         0.10891509, -0.00459933, -0.19481301, -0.01687121,  0.34054637,
#        -0.01687121, -0.18100715,  0.5215559 , -0.00459933,  0.96641064,
#        -0.49854112]
# LEAP_HOME_POSITION = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
LEAP_HOME_POSITION =  [ 0.2378,  0.0644,  0.0107, -0.0169,  0.1503,  0.0721, -0.0614, -0.3605,
        -0.0276,  0.0629, -0.0874, -0.3620,  0.2500,  0.0199,  1.3300, -0.6504]

# stacking: [ 0.1933, -0.0752,  0.0077, -0.1135,  0.2040, -0.0629, -0.1074, -0.3620,
        #  0.0844,  0.0215, -0.1565, -0.3605,  0.2991,  0.0245,  1.1167, -0.7363]

# openbottle: [ 0.0399,  0.0905,  0.0184, -0.0491,  0.0874,  0.1427, -0.0322, -0.3620,
        # -0.0920,  0.2332, -0.0337, -0.3605,  0.0690,  0.0077,  1.3821, -0.3237]
LEAP_CMD_TYPE = 'allegro'

# Jaka
JAKA_JOINT_STATE_TOPIC = '/jaka/joint_states'
JAKA_COMMANDED_JOINT_STATE_TOPIC = '/jaka/commanded_joint_states'
JAKA_EE_POSE_TOPIC = '/jaka/ee_pose'

JAKA_DOF = 6
JAKA_POSITIONS = {
    'home':[0,0,0,0,0,0],
    # 'home':[-1.5707487, 0.24192421, -1.4037328, 0.02739489, -1.8208425, -2.1729174], # TODO: change to the \pi
    # 'home':[-PI/2,-PI*2.2/180,-PI*90/180,0,-PI*90/180,-PI*124.5/180]
    # 'home': [-PI*128/180, -PI*15/180, -PI*95/180, PI*8/180, -PI*65/180, -PI*168/180]
    #'tcp_home':[-35.998, -179.06,  171.279,   -100.19/57.3,   48.86/57.3, -117.97/57.3]
    #- 'tcp_home':[-50.76120603, -169.9429,  150.56867048,   -2.99596886,   -0.06904709, -2.5355691]
    # 'tcp_home':[-50.76120603, -169.9429,  150.56867048,   -2.99596886,   -0.06904709, -2.5355691]
    #'tcp_home':[2.245301913043119, -172.0662078521393, 452.8460972484318, -0.0014055084634242037, 0.08438715452066328, -0.8852497052236575]
    # 'tcp_home':[-50.76120603, -169.9429,  180.56867048,   -2.99596886,   -0.06904709, -2.5355691] #ohter
    'tcp_home': [-50.76120603, -169.9429,  250.56867048,   -2.99596886,   -0.06904709, -2.5355691] #reorient
    # 'tcp_home': [-50.76120603, -169.9429,  150.56867048,   -2.99596886,   -0.06904709, -2.5355691] #flip
}

# [)

JAKA_IP = "192.168.130.105"
JAKA_SAFE_MOVING_TRANS = 50
SLEEP_TIME = 0.008

# Allegro
ALLEGRO_JOINT_STATE_TOPIC = '/allegroHand/joint_states'
ALLEGRO_COMMANDED_JOINT_STATE_TOPIC = '/allegroHand/commanded_joint_states'
ALLEGRO_JOINTS_PER_FINGER = 4
ALLEGRO_JOINT_OFFSETS = {
    'index': 0,
    'middle': 4,
    'ring': 8,
    'thumb': 12
}
ALLEGRO_HOME_POSITION = [ 0., -0.17453293, 0.78539816, 0.78539816, 0., -0.17453293, 0.78539816, 0.78539816, 0.08726646, -0.08726646, 0.87266463, 0.78539816, 1.04719755, 0.43633231, 0.26179939, 0.78539816]

# Kinova
KINOVA_JOINT_STATE_TOPIC = '/j2n6s300_driver/out/joint_state'

# Used for tasks
KINOVA_POSITIONS = {
    'flat': [-0.39506303664033293, 3.5573131650155982, 0.6726404554757113, 3.6574022156318287, 1.7644077385694936, 3.971040566681588],
    'slide': [-1.4591583449534833, 3.719499409085315, 1.4473843766161887, 5.551245841607734, 2.172958354550533, 0.9028881088391743],
    'opening': {
        'one_day': [5.670515511026934, 3.9037270396009633, 0.9027256560126795, 3.5419925318965904, 1.860775021562338, 5.610006360531461],
        'gatorade': [5.64553506000199, 3.57880434238032, 0.9980285400968463, 3.727822852756248, 1.53733397401416, 5.375678544640914],
        'blender': [5.651544749317862, 3.694911508005049, 0.9472571715482214, 3.6556964609536324, 1.6506910263392192, 5.4651170814257],
        'koia': [5.656182378040127, 3.7159848353947376, 0.9378061454717482, 3.6418381695891875, 1.6748827822073353, 5.486710524176281],
        'sprite': []
    }

}

# Calibration file paths
CALIBRATION_FILES_PATH = path.join(path.dirname(holodex.__path__[0]), 'calibration_files')

# Paxixni Tactile parameters
Z_TYPE = 'right' # wrong or right
FORCE_LIMIT = 30
POINT_PER_SENSOR = 15
FORCE_DIM_PER_POINT = 3
PAXINI_FINGER_PART_NAMES = {
    'tip': 'cc',
    'pulp': 'aa'
}

# this decide the order of reading tactile for each sensor board
PAXINI_FINGER_PART_INFO = {
    'tip' : b'\xcc',
    'pulp' : b'\xaa'
}

PAXINI_GROUP_INFO = {
    0 : b'\xee',
    1 : b'\xff'
}

THUMB_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB0",
    'group_id': 0,
}
INDEX_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB0",
    'group_id': 1,
}
MIDDLE_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB1",
    'group_id': 0,
}
RING_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB1",
    'group_id': 1,
}

PAXINI_LEAPHAND = {
    "thumb": THUMB_TACTILE_INFO,
    "index": INDEX_TACTILE_INFO,
    "middle": MIDDLE_TACTILE_INFO,
    "ring": RING_TACTILE_INFO
}
TACTILE_FPS = 30

PAXINI_DP_ORI_COORDS = np.array([-4.70000,  3.30000, 2.94543,
                            -4.70000,  7.80000, 2.94543,
                            -4.70000, 12.30000, 2.94543,
                            -4.70000, 16.80000, 2.94543,
                            -4.70000, 21.30000, 2.94543,
                            0.00000,  3.30000, 3.09994,
                            0.00000,  7.80000, 3.09994,
                            0.00000, 12.30000, 3.09994,
                            0.00000, 16.80000, 3.09994,
                            0.00000, 21.30000, 3.09994,
                            4.70000,  3.30000, 2.94543,
                            4.70000,  7.80000, 2.94543,
                            4.70000, 12.30000, 2.94543,
                            4.70000, 16.80000, 2.94543,
                            4.70000, 21.30000, 2.94543]).reshape(-1,3)
PAXINI_IP_ORI_COORDS = np.array([-114.60000,  4.30109, 2.97814,
                            -114.60000,  8.15109, 2.89349,
                            -114.60000, 12.18660, 2.64440,
                            -114.60000, 15.99390, 2.06277,
                            -112.35000, 21.45300, 0.09510,
                            -110.10000,  4.30109, 3.10726,
                            -110.10000,  8.15109, 3.03111,
                            -110.10000, 12.20620, 2.80133,
                            -110.10000, 16.01800, 2.25633,
                            -110.10000, 24.50520,-2.49584,
                            -105.60000,  4.30109, 2.97814,
                            -105.60000,  8.15109, 2.89349,
                            -105.60000, 12.18660, 2.64440,
                            -105.60000, 15.99390, 2.06277,
                            -107.85000, 21.45300, 0.09510]).reshape(-1,3)

# for vis
PAXINI_DP_VIS_COORDS_2D = np.array([-4.70000,  3.30000, 3.09994,
                            -4.70000,  7.80000, 3.09994,
                            -4.70000, 12.30000, 3.09994,
                            -4.70000, 16.80000, 3.09994,
                            -4.70000, 21.30000, 3.09994,
                            0.00000,  3.30000, 3.09994,
                            0.00000,  7.80000, 3.09994,
                            0.00000, 12.30000, 3.09994,
                            0.00000, 16.80000, 3.09994,
                            0.00000, 21.30000, 3.09994,
                            4.70000,  3.30000, 3.09994,
                            4.70000,  7.80000, 3.09994,
                            4.70000, 12.30000, 3.09994,
                            4.70000, 16.80000, 3.09994,
                            4.70000, 21.30000, 3.09994]).reshape(-1,3)
PAXINI_DP_VIS_COORDS_2D -= np.mean(PAXINI_DP_VIS_COORDS_2D,0)
PAXINI_DP_VIS_COORDS_2D /= np.max(abs(PAXINI_DP_VIS_COORDS_2D))
PAXINI_DP_VIS_COORDS_2D /= 2

PAXINI_DP_VIS_COORDS_3D = np.array([-4.70000,  3.30000, 3.09994,
                            -4.70000,  7.80000, 3.09994,
                            -4.70000, 12.30000, 3.09994,
                            -4.70000, 16.80000, 3.09994,
                            -4.70000, 21.30000, 3.09994,
                            0.00000,  3.30000, 3.09994,
                            0.00000,  7.80000, 3.09994,
                            0.00000, 12.30000, 3.09994,
                            0.00000, 16.80000, 3.09994,
                            0.00000, 21.30000, 3.09994,
                            4.70000,  3.30000, 3.09994,
                            4.70000,  7.80000, 3.09994,
                            4.70000, 12.30000, 3.09994,
                            4.70000, 16.80000, 3.09994,
                            4.70000, 21.30000, 3.09994]).reshape(-1,3)

PAXINI_IP_VIS_COORDS_2D = np.array([[-4.5    ,  4.30109,  2.97814],
                            [-4.5    ,  8.15109,  2.89349],
                            [-4.5    , 12.1866 ,  2.6444 ],
                            [-4.5    , 15.9939 ,  2.06277],
                            [-2.25   , 21.453  ,  0.0951 ],
                            [ 0.     ,  4.30109,  3.10726],
                            [ 0.     ,  8.15109,  3.03111],
                            [ 0.     , 12.2062 ,  2.80133],
                            [ 0.     , 16.018  ,  2.25633],
                            [ 0.     , 24.5052 , -2.49584],
                            [ 4.5    ,  4.30109,  2.97814],
                            [ 4.5    ,  8.15109,  2.89349],
                            [ 4.5    , 12.1866 ,  2.6444 ],
                            [ 4.5    , 15.9939 ,  2.06277],
                            [ 2.25   , 21.453  ,  0.0951 ]])
PAXINI_IP_VIS_COORDS_2D -= np.mean(PAXINI_IP_VIS_COORDS_2D,0)
PAXINI_IP_VIS_COORDS_2D /= np.max(abs(PAXINI_IP_VIS_COORDS_2D))
PAXINI_IP_VIS_COORDS_2D /= 2

PAXINI_IP_VIS_COORDS_3D = np.array([[-4.5    ,  4.30109,  2.97814],
                            [-4.5    ,  8.15109,  2.89349],
                            [-4.5    , 12.1866 ,  2.6444 ],
                            [-4.5    , 15.9939 ,  2.06277],
                            [-2.25   , 21.453  ,  0.0951 ],
                            [ 0.     ,  4.30109,  3.10726],
                            [ 0.     ,  8.15109,  3.03111],
                            [ 0.     , 12.2062 ,  2.80133],
                            [ 0.     , 16.018  ,  2.25633],
                            [ 0.     , 24.5052 , -2.49584],
                            [ 4.5    ,  4.30109,  2.97814],
                            [ 4.5    ,  8.15109,  2.89349],
                            [ 4.5    , 12.1866 ,  2.6444 ],
                            [ 4.5    , 15.9939 ,  2.06277],
                            [ 2.25   , 21.453  ,  0.0951 ]])

# Realsense Camera parameters
NUM_CAMS = 3
CAM_FPS = 30
WIDTH = 1280
HEIGHT = 720
PROCESSING_PRESET = 1 # High accuracy post-processing mode
VISUAL_RESCALE_FACTOR = 2
RECORD_FPS = 10

# Mediapipe detector
# ROS Topics
MP_RGB_IMAGE_TOPIC = "/mediapipe/original/color_image"
MP_DEPTH_IMAGE_TOPIC = "/mediapipe/original/depth_image"

MP_KEYPOINT_TOPIC = "/mediapipe/predicted/keypoints"
MP_PRED_BOOL_TOPIC = "/mediapipe/predicted/detected_boolean"

MP_PRED_RGB_IMAGE_TOPIC = "/mediapipe/predicted/color_image"
MP_PRED_DEPTH_IMAGE_TOPIC = "/mediapipe/predicted/depth_image"

MP_HAND_TRANSFORM_COORDS_TOPIC = "/mediapipe/predicted/transformed_keypoints"
MP_HAND_TRANSFORM_COORDS_TOPIC = "/mediapipe/predicted/transformed_keypoints"

# File paths
MP_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'mp_bounds.npy')

# Joint information
MP_NUM_KEYPOINTS = 11
MP_THUMB_BOUND_VERTICES = 4

MP_OG_KNUCKLES = [1, 5, 9, 13, 17]
MP_OG_FINGERTIPS = [4, 8, 12, 16, 20]

MP_JOINTS = {
    'metacarpals': [1, 2, 3, 4, 5],
    'knuckles': [2, 3, 4, 5],
    'thumb': [1, 6],
    'index':[2, 7],
    'middle': [3, 8],
    'ring': [4, 9],
    'pinky': [5, 10] 
}


MP_VIEW_LIMITS = {
    'x_limits': [-0.12, 0.12],
    'y_limits': [-0.02, 0.2],
    'z_limits': [0, 0.06]
}

# Other params
MP_PROCESSING_PRESET = 2 # Hands post-processing mode - Hands mode
PRED_CONFIDENCE = 0.95
MAX_NUM_HANDS = 1
MP_FREQ = 30

# Leapmotion detector
# ROS Topics names
LP_HAND_KEYPOINT_TOPIC = "/leapmotion/hand_keypoints"
LP_ARM_KEYPOINT_TOPIC = "/leapmotion/arm_keypoints"

LP_HAND_TRANSFORM_COORDS_TOPIC = "/leapmotion/transformed_hand_keypoints"
LP_ARM_TRANSFORM_COORDS_TOPIC = "/leapmotion/transformed_arm_keypoints"

# Joint information
LP_NUM_KEYPOINTS = 21
LP_ARM_NUM_KEYPOINTS = 4

LP_HAND_VISULIZATION_LINKS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

LP_JOINTS = {
    'metacarpals': [1, 5, 9, 13, 17],
    'knuckles': [5, 9, 13, 17],
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

LP_VIEW_LIMITS = {
    'x_limits': [-0.2, 0.2],
    'y_limits': [-0.2, 0.2],
    'z_limits': [-0.2, 0.2]
}

# Other params
LP_FREQ = 30

LP_TO_ROBOT = np.array([-1, 0, 0,
                                 0, 0, 1,
                                 0, 1, 0]).reshape(3, 3)

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

LP_WORKSPACE_SCALE = 2

# VR detector 
# ROS Topic names
VR_RIGHT_ARM_KEYPOINTS_TOPIC = '/OculusVR/right_arm_keypoints'
VR_RIGHT_HAND_KEYPOINTS_TOPIC = '/OculusVR/right_hand_keypoints'
VR_LEFT_HAND_KEYPOINTS_TOPIC = '/OculusVR/left_hand_keypoints'

VR_RIGHT_ARM_TRANSFORM_COORDS_TOPIC = '/OculusVR/transformed_right_arm'
VR_RIGHT_TRANSFORM_COORDS_TOPIC = "/OculusVR/transformed_right"
VR_LEFT_TRANSFORM_DIR_TOPIC = "/OculusVR/left_dir_vectors"

# File paths
VR_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'vr_thumb_bounds.npy')
VR_DISPLAY_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'vr_thumb_plot_bounds.npy')
VR_2D_PLOT_SAVE_PATH = path.join(CALIBRATION_FILES_PATH, 'oculus_hand_2d_plot.jpg')

# Joint Information
OCULUS_NUM_KEYPOINTS = 24
OCULUS_ARM_NUM_KEYPOINTS = 3
VR_THUMB_BOUND_VERTICES = 8

OCULUS_JOINTS = {
    'metacarpals': [2, 6, 9, 12, 15],
    'knuckles': [6, 9, 12, 16],
    'thumb': [2, 3, 4, 5, 19],
    'index': [6, 7, 8, 20],
    'middle': [9, 10, 11, 21],
    'ring': [12, 13, 14, 22],
    'pinky': [15, 16, 17, 18, 23]
}

OCULUS_VIEW_LIMITS = {
    'x_limits': [-0.04, 0.04],
    'y_limits': [-0.02, 0.25],
    'z_limits': [-0.04, 0.04]
}

# Other params
VR_FREQ = 60 #60 original

LEFT_TO_RIGHT = np.array([1, 0, 0,
                          0, -1, 0,
                          0, 0, 1]).reshape(3, 3)

ARM_POS_SCALE = 580
ARM_ORI_SCALE = np.pi
TACTILE_RAW_DATA_SCALE = 60
ARM_JOINT_LOWER_LIMIT = torch.tensor([-6.28, -2.09, -2.27, -6.28, -2.09, -6.28])
ARM_JOINT_UPPER_LIMIT = torch.tensor([6.28, 2.09, 2.27, 6.28, 2.09, 6.28])
HAND_JOINT_LOWER_LIMIT = torch.tensor([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -1.047, -0.314,
                                       -0.506, -0.366, -0.349, -0.47, -1.2, -1.34])
HAND_JOINT_UPPER_LIMIT = torch.tensor([1.047, 2.23, 1.885, 2.042, 1.047, 2.23, 1.885, 2.042, 1.047, 2.23, 1.885, 2.042,
                                       2.094, 2.443, 1.9, 1.88])


# Hamer detector
# ROS Topics names
HAMER_HAND_KEYPOINT_TOPIC = "/hamer/hand_keypoints"
HAMER_ARM_KEYPOINT_TOPIC = "/hamer/arm_keypoints"
HAMER_HAND_TRANSFORM_COORDS_TOPIC = "/hamer/transformed_hand_keypoints"
HAMER_ARM_TRANSFORM_COORDS_TOPIC = "/hamer/transformed_arm_keypoints"

HAMER_DESIRED_COORDS_TOPIC = "/hamer/desired_joints"
HAMER_RETARGETED_COORDS_TOPIC = "/hamer/retargeted_joints"

HAMER_HAND_POS_TOPIC = "/hamer/hand_position"
HAMER_HAND_VEL_TOPIC = "/hamer/hand_velocity"
HAMER_HAND_TORQUE_TOPIC = "/hamer/hand_torque"

IMU_ROT_TOPIC = "/hamer/imu_rpy"

OBJECT_POSE_TOPIC = "/object/Pose"

# Joint information
HAMER_NUM_KEYPOINTS = 21
HAMER_ARM_NUM_KEYPOINTS = 3

# Other params
HAMER_FREQ = 30
