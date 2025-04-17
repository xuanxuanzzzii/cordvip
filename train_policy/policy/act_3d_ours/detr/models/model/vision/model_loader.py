import pathlib
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import time

models_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(
    str(models_path / "models" / "pointnet" / "Pointnet_Pointnet2_pytorch" / "models")
)
from co_diffusion_policy.model.vision.pointnet2_cls_ssg import (
    get_model as Pointnet2,
)


class Pointnet2Enc(Pointnet2):
    def __init__(self):
        super(Pointnet2Enc, self).__init__(normal_channel=False)
        # self.feature_dim = 256
        self.fc_out = nn.Linear(256,1024)
    def forward(self, xyz):
        if len(xyz.shape) == 2:
            xyz = xyz.unsqueeze(0)
        B, N, C = xyz.shape
        xyz = xyz[:, :, :3]
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        start_time = time.perf_counter()
        l1_xyz, l1_points = self.sa1(xyz, norm)
        sa1_time = time.perf_counter() - start_time
        print(f"sa1 operation time: {sa1_time:.4f} seconds")

        start_time = time.perf_counter()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        sa2_time = time.perf_counter() - start_time
        print(f"sa2 operation time: {sa2_time:.4f} seconds")
        
        start_time = time.perf_counter()
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        sa3_time = time.perf_counter() - start_time
        print(f"sa3 operation time: {sa3_time:.4f} seconds")

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc_out(x)
        return x


# if __name__ == "__main__":
#     from lift3d.helpers.common import Logger

#     device = "cuda"
#     point_clouds = torch.randn([4, 4096, 3]).to(device)

#     # Test PointNet
#     pointnet = PointnetEnc().to(device)
#     Logger.log_info("Test PointNet")
#     embedding = pointnet.forward(point_clouds)
#     assert embedding.shape[1] == pointnet.feature_dim
#     Logger.log_info(f"feature dimension: {pointnet.feature_dim}")

#     # Test PointNet2
#     pointnet2 = Pointnet2Enc().to(device)
#     Logger.log_info("Test PointNet2")
#     embedding = pointnet2.forward(point_clouds)
#     assert embedding.shape[1] == pointnet2.feature_dim
#     Logger.log_info(f"feature dimension: {pointnet2.feature_dim}")
