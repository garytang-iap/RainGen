import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .knn import *
import math

from .conv_uno import UNO
from .sparse_conv_block import SparseConvResBlock
from .sparse_conv_block import convert_to_backend_form, convert_to_backend_form_like, \
    calculate_norm, get_features_from_backend_form, get_normalising_conv

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class EnhancedSparseUNet(nn.Module):
    def __init__(self, channels=1, condition_dim=7, nf=64, time_emb_dim=256, img_size=128, 
                 num_conv_blocks=3, knn_neighbours=3, uno_res=64, uno_mults=(1,2,4,8), 
                 out_channels=None, conv_type="conv", depthwise_sparse=True, kernel_size=7, 
                 backend="torch_dense", optimise_dense=True, blocks_per_level=(2,2,2,2), 
                 attn_res=[16,8], dropout_res=16, dropout=0.1, uno_base_nf=64, cond_drop_prob=0.0):
        super().__init__()
        
        # 基础参数
        self.backend = backend
        self.default_img_size = img_size # 记录默认值
        self.default_uno_res = uno_res   # 记录默认值
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size
        self.condition_dim = condition_dim
        self.cond_drop_prob = cond_drop_prob 
        
        input_channels = channels + condition_dim
        self.linear_in = nn.Linear(input_channels, nf)
        self.linear_out = nn.Linear(nf, out_channels if out_channels is not None else channels)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 🔥 兼容性保留：虽然我们想动态生成，但为了不破坏 load_state_dict，保留这个 buffer
        # 实际 forward 中我们会忽略它
        uno_coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=uno_res) for _ in range(2)]))
        uno_coords = rearrange(uno_coords, 'c h w -> () (h w) c')
        self.register_buffer("uno_coords", uno_coords)

        self.normalising_conv = get_normalising_conv(kernel_size=kernel_size, backend=backend)

        self.down_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, 
                time_emb_dim=time_emb_dim, z_dim=None,
                depthwise=depthwise_sparse, backend=backend
            ))

        self.uno_linear_in = nn.Linear(nf, uno_base_nf)
        self.uno_linear_out = nn.Linear(uno_base_nf, nf)
        
        self.uno = UNO(uno_base_nf, uno_base_nf, width=uno_base_nf, mults=uno_mults, 
                       blocks_per_level=blocks_per_level, time_emb_dim=time_emb_dim, 
                       z_dim=None, conv_type=conv_type, res=uno_res,
                       attn_res=attn_res, dropout_res=dropout_res, dropout=dropout)

        self.up_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.up_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, skip_dim=nf, 
                time_emb_dim=time_emb_dim, z_dim=None,
                depthwise=depthwise_sparse, backend=backend
            ))

    def _get_dynamic_uno_coords(self, batch_size, res, device):
        """🔥 动态生成 UNO 网格坐标 (0到1)，替代 buffer"""
        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, steps=res, device=device),
            torch.linspace(0, 1, steps=res, device=device),
            indexing='ij'
        )) 
        coords = rearrange(coords, 'c h w -> () (h w) c')
        coords = coords.repeat(batch_size, 1, 1)
        return coords

    def knn_interpolate_to_grid(self, x, coords, current_uno_res):
        """🔥 使用动态分辨率进行 KNN 插值"""
        # 动态生成目标网格 (匹配 current_uno_res)
        grid_coords = self._get_dynamic_uno_coords(x.size(0), current_uno_res, x.device)
        
        interpolated_features = knn_interpolate(
            features=x,
            coords=coords,
            grid_coords=grid_coords,
            k=self.knn_neighbours
        )
        return interpolated_features

    def forward(self, x, t, condition=None, sample_lst=None, coords=None, 
                cond_drop_prob=None, img_size=None, uno_res=None):
        """
        新增参数:
        - cond_drop_prob: 显式控制 dropout, 设为 1.0 为无条件, 0.0 为有条件
        - img_size: 动态控制输入图像尺寸 (Sparse Conv用)
        - uno_res: 动态控制中间层分辨率 (UNO用)
        """
        batch_size, num_points, _ = x.shape
        device = x.device
        
        # 🔥 1. 确定当前尺寸 (优先使用传入参数，否则用默认)
        curr_img_size = img_size if img_size is not None else self.default_img_size
        curr_uno_res = uno_res if uno_res is not None else self.default_uno_res

        if x.dtype == torch.float16 and t.dtype == torch.float32:
            t = t.half()
        
        # 🔥 2. 改进的 CFG 逻辑 (兼容旧权重 -10.0)
        # 如果显式传入了 cond_drop_prob，就用它；否则看 self.training
        if cond_drop_prob is None:
            drop_prob = self.cond_drop_prob if self.training else 0.0
        else:
            drop_prob = cond_drop_prob

        # 使用训练时的 -10.0 逻辑，不引入新参数
        if drop_prob > 0. and condition is not None:
            keep_mask = prob_mask_like((batch_size,), 1. - drop_prob, device=device).view(batch_size, 1, 1)
            # 使用 -10.0 填充，保持和训练一致
            null_cond = torch.full_like(condition, -10.0)
            condition = torch.where(keep_mask, condition, null_cond)
        
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)
      
        x = self.linear_in(x)
        t = self.time_mlp(t)

        # 3. 动态生成输入坐标 (如果未提供)
        if coords is None:
            # 假设 sample_lst 是基于 curr_img_size 的索引
            coords_full = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, steps=curr_img_size, device=device),
                torch.linspace(0, 1, steps=curr_img_size, device=device),
                indexing='ij'
            ))
            coords_full = rearrange(coords_full, 'c h w -> () (h w) c')
            coords_full = repeat(coords_full, "() ... -> b ...", b=batch_size)
            if sample_lst is not None:
                coords = torch.gather(coords_full, 1, sample_lst.unsqueeze(2).repeat(1, 1, 2)).contiguous()
            else:
                coords = coords_full

        # Sparse Encoder (传入 curr_img_size)
        x = convert_to_backend_form(x, sample_lst, curr_img_size, backend=self.backend)
        backend_tensor = x
        norm = calculate_norm(self.normalising_conv, backend_tensor, sample_lst, curr_img_size, batch_size, backend=self.backend)

        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, norm=norm)
            downs.append(x)

        # Dense UNO
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.uno_linear_in(x)
        
        # 🔥 动态插值到 curr_uno_res
        x = self.knn_interpolate_to_grid(x, coords, curr_uno_res)
        x = rearrange(x, "b (h w) c -> b c h w", h=curr_uno_res)

        x = self.uno(x, t) # UNO 内部是全卷积，支持动态尺寸

        # 🔥 Grid Sample 需要 [-1, 1] 坐标，这里假设 coords 是 [0, 1]
        # 如果你的 Dataset 输出是 [-1, 1]，这里 grid_sample 直接用 coords，不需要转换
        # 但 knn_interpolate 最好用 [0, 1]。
        # 为兼容性，这里使用和原来一致的逻辑：
        # 原代码：x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear')
        # 我们保持不变。
        x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear', align_corners=False)
        x = rearrange(x, "b c l () -> b l c")
        
        x = self.uno_linear_out(x)
        x = convert_to_backend_form_like(x, backend_tensor, sample_lst=sample_lst, 
                                       img_size=curr_img_size, backend=self.backend)

        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, skip=skip, norm=norm)

        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.linear_out(x)

        return x