# file: your_project/lib/utils/knn_utils.py

import torch
from einops import rearrange

@torch.no_grad()
def knn_interpolate(
    features: torch.Tensor,
    coords: torch.Tensor,
    grid_coords: torch.Tensor,
    k: int,
    inverse_dist_weighting: bool = True,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    使用纯PyTorch实现KNN插值，将稀疏点特征插值到规则网格上。
    不依赖于pytorch3d。

    Args:
        features (torch.Tensor): 稀疏点的特征，形状为 [B, S, D]。
        coords (torch.Tensor): 稀疏点的坐标，形状为 [B, S, 2]。
        grid_coords (torch.Tensor): 目标网格的坐标，形状为 [B, N, 2]，其中 N = H * W。
        k (int): K近邻的数量。
        inverse_dist_weighting (bool): 是否使用反距离加权。
        epsilon (float): 用于防止除以零的小常数。

    Returns:
        torch.Tensor: 插值后的网格特征，形状为 [B, N, D]。
    """
    B, S, D = features.shape
    B, N, _ = grid_coords.shape

    # 1. 计算所有网格点到所有稀疏点的距离矩阵
    # 扩展维度以利用广播机制
    grid_coords_exp = grid_coords.unsqueeze(2)      # [B, N, 1, 2]
    sparse_coords_exp = coords.unsqueeze(1)         # [B, 1, S, 2]

    # 计算平方距离
    dist_sq = torch.sum((grid_coords_exp - sparse_coords_exp)**2, dim=-1) # [B, N, S]

    # 2. 找到每个网格点最近的 K 个稀疏点
    # torch.topk 找的是最大的值，所以我们对负的距离矩阵操作
    # topk() 返回 (values, indices)
    _, topk_indices = torch.topk(-dist_sq, k=k, dim=2) # [B, N, K]

    # 3. 收集最近邻的特征和坐标
    # topk_indices 需要扩展维度以匹配特征和坐标的维度
    # .detach()可以防止不必要的梯度计算，虽然整个函数在 no_grad() 下
    topk_indices_detached = topk_indices.detach()
    
    # 扩展索引以用于gather
    expanded_indices_feat = topk_indices_detached.unsqueeze(-1).expand(-1, -1, -1, D)
    expanded_indices_coord = topk_indices_detached.unsqueeze(-1).expand(-1, -1, -1, 2)
    
    # 使用 gather 获取 K 个最近邻的特征和坐标
    # B, N, S, D -> B, N, K, D
    neighbour_features = torch.gather(features.unsqueeze(1).expand(-1, N, -1, -1), 2, expanded_indices_feat)
    # B, N, S, 2 -> B, N, K, 2
    neighbour_coords = torch.gather(coords.unsqueeze(1).expand(-1, N, -1, -1), 2, expanded_indices_coord)
    
    # 4. 计算权重并进行加权平均
    if inverse_dist_weighting:
        # 计算网格点到 K 个最近邻的距离
        dist_to_neighbours_sq = torch.sum((grid_coords.unsqueeze(2) - neighbour_coords)**2, dim=-1) # [B, N, K]
        
        # 反距离加权
        weights = 1.0 / (torch.sqrt(dist_to_neighbours_sq) + epsilon) # [B, N, K]
        
        # 归一化权重
        weights_sum = weights.sum(dim=-1, keepdim=True)
        weights = weights / (weights_sum + epsilon) # [B, N, K]
        
        # 加权平均
        # weights.unsqueeze(-1): [B, N, K, 1]
        interpolated_features = torch.sum(neighbour_features * weights.unsqueeze(-1), dim=2) # [B, N, D]
    else:
        # 简单平均
        interpolated_features = torch.mean(neighbour_features, dim=2) # [B, N, D]
        
    return interpolated_features.to(features.dtype)