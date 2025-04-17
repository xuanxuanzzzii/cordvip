import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from typing import Dict
from CordViP.model.vision.transformer import Transformer
from CordViP.model.vision.pointnet_extractor import PointNetEncoderXYZRGB, PointNetEncoderXYZ
import os

from CordViP.model.vision.pointnet2_encoder import PointNet2EncoderXYZ
from CordViP.model.vision.pointnext_encoder import PointNextEncoderXYZ

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20):
    idx = knn(x, k=k)
    batch_size, num_points, _ = idx.size()
    _, num_dims, _ = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

def create_encoder_network(emb_dim, pretrain=None, device=torch.device('cpu')) -> nn.Module:
    encoder = DGCNNEncoder(emb_dim=emb_dim)
    if pretrain is not None:
        print(f"******** Load embedding network pretrain from <{pretrain}> ********")
        encoder.load_state_dict(
            torch.load(
                os.path.join(ROOT_DIR, f"ckpt/pretrain/{pretrain}"),
                map_location=device
            )
        )
    return encoder

class PointROEncoder(nn.Module):
    """
    Create a observation encoder

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """
    def __init__(self, 
                observation_space: Dict,
                encoder_emb_dim=1024, 
                encoder_pretrain=None,
                use_pc_color=False,
                pointnet_type='pointnet',
                pointcloud_encoder_cfg=None,
                robot_embedding_dim=64,
                ):
        super().__init__()
        self.encoder_emb_dim = encoder_emb_dim
        self.robot_embedding_dim = robot_embedding_dim

        self.state_key = 'joint_state'
        self.robot_point_cloud_key = 'hand_point_cloud'
        self.object_point_cloud_key = 'object_point_cloud'

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.encoder_pretrain = encoder_pretrain
        
        print ("self.pointnet_type",self.pointnet_type)
        if self.pointnet_type == "pointnet":
            print ("pointnet")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.encoder_robot = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                self.encoder_object = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.encoder_robot = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
                self.encoder_object = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif self.pointnet_type == "pointnet2":
            print ("pointnet++")
            self.encoder_robot = PointNet2EncoderXYZ(**pointcloud_encoder_cfg)
            self.encoder_object = PointNet2EncoderXYZ(**pointcloud_encoder_cfg)
        elif self.pointnet_type == "point_next":
            print ("point_next")
            self.encoder_robot = PointNextEncoderXYZ(**pointcloud_encoder_cfg)
            self.encoder_object = PointNextEncoderXYZ(**pointcloud_encoder_cfg)
        elif self.pointnet_type == "dgcnn":
            if use_pc_color:
                cprint("dgcnn doesn't support color point cloud")
                pass
            else:
                self.encoder_robot = create_encoder_network(emb_dim=encoder_emb_dim, pretrain=encoder_pretrain)
                self.encoder_object = create_encoder_network(emb_dim=encoder_emb_dim)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        self.transformer_robot = Transformer(emb_dim=encoder_emb_dim)
        self.transformer_object = Transformer(emb_dim=encoder_emb_dim)

        self.hand_projection = nn.Linear(16,robot_embedding_dim)
        self.arm_projection = nn.Linear(6,robot_embedding_dim)

        self.attn_hand = Transformer(emb_dim=robot_embedding_dim)
        self.attn_arm = Transformer(emb_dim=robot_embedding_dim)
        

    def forward(self, observations: Dict) -> torch.Tensor:
        robot_points = observations[self.robot_point_cloud_key] # bs, h, 1024, 3
        object_points = observations[self.object_point_cloud_key] # bs, h, 1024, 3
        
        joint_state = observations[self.state_key] # bs, h, 22
        batch_size = robot_points.shape[0]

        # robot_points = robot_points.view(-1, *robot_points.shape[2:]) # bs*h, 1024, 3
        # object_points = object_points.view(-1, *object_points.shape[2:]) # bs*h, 1024, 3
        robot_points = robot_points.reshape(-1, *robot_points.shape[2:])
        object_points = object_points.reshape(-1, *object_points.shape[2:])
        robot_pc_embedding = self.encoder_robot(robot_points) # bs*h, encoder_emb_dim
        object_pc_embedding = self.encoder_object(object_points) # bs*h, encoder_emb_dim
        # print("robot_pc_embedding",robot_pc_embedding.shape)
        robot_pc_embedding = robot_pc_embedding.view(batch_size, -1, *robot_pc_embedding.shape[1:])
        object_pc_embedding = object_pc_embedding.view(batch_size, -1, *object_pc_embedding.shape[1:])

        if self.encoder_pretrain is not None:
            robot_pc_embedding = robot_pc_embedding.detach()
        
        transformer_robot_outputs = self.transformer_robot(robot_pc_embedding, object_pc_embedding) # bs, h, encoder_emb_dim
        transformer_object_outputs = self.transformer_object(object_pc_embedding, robot_pc_embedding) # bs, h, encoder_emb_dim
        robot_embedding_tf = robot_pc_embedding + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_pc_embedding + transformer_object_outputs["src_embedding"]
        
        # hand and arm
        joint_hand = joint_state[:, :, :16]  # hand  torch.Size([bs, h, 16])
        joint_arm = joint_state[:, :, 16:]  # arm  torch.Size([bs, h, 6])

        joint_hand = self.hand_projection(joint_hand) # (bs, h, 64)
        joint_arm = self.arm_projection(joint_arm) # (bs, h, 64)
        
        # hand
        outputs = self.attn_hand(joint_hand, joint_arm)
        hand_embedding = outputs["src_embedding"] # (bs, h, 64)
        hand_embedding_con = joint_hand + hand_embedding # (bs, h,64)
        
        # arm
        outputs = self.attn_arm(joint_arm, joint_hand)
        arm_embedding = outputs["src_embedding"] # (bs, h, 64)
        arm_embedding_con = joint_arm + arm_embedding # (bs, h, 64)

        return robot_embedding_tf, object_embedding_tf, hand_embedding_con, arm_embedding_con

    def output_shape(self):
        return self.robot_embedding_dim * 2 + self.encoder_emb_dim * 2

class DGCNNEncoder(nn.Module):
    """
    The implementation is based on the DGCNN model
    (https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py#L88),
    and https://github.com/r-pad/taxpose/blob/0c4298fa0486fd09e63bf24d618a579b66ba0f18/third_party/dcp/model.py#L282.

    Further explanation can be found in Appendix F.1 of https://arxiv.org/pdf/2410.01702.
    """

    def __init__(self, emb_dim=512):
        super(DGCNNEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(1536, emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
        B, _, N = x.size()

        x = get_graph_feature(x, k=32)  # (B, 6, N, K)

        x = self.conv1(x)  # (B, 64, N, K)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = self.conv2(x)  # (B, 64, N, K)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = self.conv3(x)  # (B, 128, N, K)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        x = self.conv4(x)  # (B, 256, N, K)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        x = self.conv5(x)  # (B, 512, N, K)
        x5 = x.max(dim=-1, keepdim=False)[0]  # (B, 512, N)

        global_feat = x5.mean(dim=-1, keepdim=True).repeat(1, 1, N)  # (B, 512, 1) -> (B, 512, N)

        x = torch.cat((x1, x2, x3, x4, x5, global_feat), dim=1)  # (B, 1536, N)
        x = self.conv6(x).view(B, -1, N)  # (B, 512, N)

        return x.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)

class CvaeEncoder(nn.Module):
    """
    The implementation is based on the DGCNN model
    (https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py#L88).

    The only modification made is to enable the input to include additional features.
    """

    def __init__(self, emb_dims, output_channels, feat_dim=0):
        super(CvaeEncoder, self).__init__()
        self.feat_dim = feat_dim

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6 + feat_dim, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, D, N = x.size()
        x_k = get_graph_feature(x[:, :3, :])  # B, 6, N, K
        x_feat = x[:, 3:, :].unsqueeze(-1).repeat(1, 1, 1, 20) if self.feat_dim != 0 else None  # K = 20
        x = torch.cat([x_k, x_feat], dim=1) if self.feat_dim != 0 else x_k  # (B, 6 + feat_dim, N, K)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)[..., 0]

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x  # (B, output_channels)


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     obs_dict = {"object_point_cloud": [512, 3], "hand_point_cloud": [512, 3], "joint_state": [22]}
#     encoder = PointROEncoder(obs_dict).to(device)
#     obs = {"object_point_cloud": torch.zeros(32, 4, 512, 3).to(device), "hand_point_cloud": torch.zeros(32, 4, 512, 3).to(device), "joint_state": torch.zeros(32, 4, 22).to(device)}
#     robot_embedding_tf, object_embedding_tf, hand_embedding_con, arm_embedding_con = encoder(obs)
#     print("robot_embedding_tf:", robot_embedding_tf.shape)
#     print("object_embedding_tf:", object_embedding_tf.shape)
#     print("hand_embedding_con:", hand_embedding_con.shape)
#     print("arm_embedding_con:", arm_embedding_con.shape)