import torch
import torch.nn.functional as F

def pointcloud_distance(pred_pc, target_pc, type='chamfer'):
    if type == 'chamfer':
        return chamfer_distance(pred_pc, target_pc)
    elif type == 'emd':
        return earth_mover_distance(pred_pc, target_pc)
    elif type == 'mse':
        return F.mse_loss(pred_pc, target_pc)
    elif type == 'l1':
        return F.l1_loss(pred_pc, target_pc)
    else:
        raise NotImplementedError('Unknown distance type: {}'.format(type))
    
def chamfer_distance(pc1, pc2):
    """
    Calculate Chamfer Distance between two point clouds.
    
    Parameters:
    - pc1: point cloud 1, shape (B, T, N, 3)
    - pc2: point cloud 2, shape (B, T, N, 3)
    
    Returns:
    - distance: Chamfer Distance, scalar
    """
    
    B, T, N, _ = pc1.shape
    assert pc1.shape == pc2.shape
    assert pc1.shape[-1] == 3
    
    pc1 = pc1.reshape(B * T, N, 3)
    pc2 = pc2.reshape(B * T, N, 3)
    
    pc1_expand = pc1.unsqueeze(2)
    pc2_expand = pc2.unsqueeze(1)
    
    # Pairwise distance
    dists = torch.sum((pc1_expand - pc2_expand) ** 2, dim=3)
    
    # Distance from pc1 to pc2
    min_dists_pc1 = torch.min(dists, dim=2)[0]
    chamfer1 = torch.mean(min_dists_pc1, dim=1)
    
    # Distance from pc2 to pc1
    min_dists_pc2 = torch.min(dists, dim=1)[0]
    chamfer2 = torch.mean(min_dists_pc2, dim=1)
    
    return torch.mean(chamfer1 + chamfer2)

def earth_mover_distance(pc1, pc2):
    """
    Calculate Earth Mover's Distance between two point clouds.
    
    Parameters:
    - pc1: point cloud 1, shape (B, T, N, 3)
    - pc2: point cloud 2, shape (B, T, N, 3)
    
    Returns:
    - distance: Earth Mover's Distance, scalar
    """
    
    B, T, N, _ = pc1.shape
    
    pc1 = pc1.view(B * T, N, 3)
    pc2 = pc2.view(B * T, N, 3)
    
    pc1_expand = pc1.unsqueeze(2)
    pc2_expand = pc2.unsqueeze(1)
    
    # Pairwise distance
    dists = torch.sum((pc1_expand - pc2_expand) ** 2, dim=3)
    
    # Solve the linear sum assignment problem to find the best match
    # This is just an approximation using softmin
    match = F.softmax(-dists, dim=2)
    emd = torch.sum(dists * match, dim=[1, 2])
    
    return torch.mean(emd)
