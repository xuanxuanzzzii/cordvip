import torch
import torch.nn as nn
from dexenv.models.voxel_model import VoxelModel
from dexenv.utils.minkowski_utils import batched_coordinates_array
from dexenv.utils.voxel_utils import create_input_batch
from dexenv.utils.torch_utils import unique

class VoxelCNN(nn.Module):
    """
    from Tao Chen's Visual Dexterity
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super().__init__()
        self.body_net = VoxelModel(embed_dim=256,
                              out_features=out_channels,
                              batch_norm=True,
                              act='relu',
                              channel_groups=[32, 64, 128, 256],
                              in_channels=1,
                              layer_norm=False)
        self.color_channels = 1 # we do not use color for this network, same as Tao Chen's code
        self.act = nn.GELU()
        

    
    def flat_batch_time(self, x):
        return x.view(x.shape[0] * x.shape[1], *x.shape[2:])

    def unflat_batch_time(self, x, b, t):
        return x.view(b, t, *x.shape[1:])
    
    @torch.no_grad()
    def convert_to_sparse_tensor(self, coords, color=None):
        b = coords.shape[0]
        t = coords.shape[1]
        flat_coords = self.flat_batch_time(coords)


        coordinates_batch = batched_coordinates_array(flat_coords, device=coords.device)
        coordinates_batch, uindices = unique(coordinates_batch, dim=0)
        features_batch = torch.full((coordinates_batch.shape[0], self.color_channels),
                                    0.5, device=coordinates_batch.device)
        batch = {
            "coordinates": coordinates_batch,
            "features": features_batch,
        }
        input_batch_sparse = create_input_batch(batch, device=coords.device,
                                                quantization_size=None,
                                                speed_optimized=True,
                                                quantization_mode='random')
        return input_batch_sparse, b, t


    def forward(self, x):
        """
        x: (B, N, 3)
        """
        coords = x.unsqueeze(1)
        color = None
        x_sparse_tensor, b, t = self.convert_to_sparse_tensor(coords=coords, color=color)
        voxel_features = self.body_net(x_sparse_tensor)
        voxel_features = self.act(voxel_features)
        return voxel_features