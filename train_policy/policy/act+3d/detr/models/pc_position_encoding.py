import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class PcPosEnc(nn.Module):
    def __init__(self, d_model, k):
        super().__init__()
        self.k = k
        self.delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, pc):
        dist = square_distance(pc, pc)
        knn_idx = dist.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(pc, knn_idx)
        pos_enc = self.delta(pc[:, :, None] - knn_xyz)
        return pos_enc