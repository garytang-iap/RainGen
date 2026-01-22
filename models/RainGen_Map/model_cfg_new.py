import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
#from pytorch3d.ops import knn_points, knn_gather
from .knn import *
import math
import warnings
# ... (前面的 import 保持不变)
from einops import rearrange
from .conv_uno import UNO, UNOEncoder
from .sparse_conv_block import SparseConvResBlock
from .sparse_conv_block import convert_to_backend_form, convert_to_backend_form_like, \
    calculate_norm, get_features_from_backend_form, get_normalising_conv
def coords_pix_to_virtual_sample_lst(coords_pix_norm, img_size, virtual_scale):
    # coords_pix_norm: [B,L,2] in [-1,1], (x,y)
    vsize = int(img_size * virtual_scale)
    x = coords_pix_norm[..., 0]
    y = coords_pix_norm[..., 1]
    xv = ((x + 1) * 0.5 * (vsize - 1)).round().clamp(0, vsize - 1).long()
    yv = ((y + 1) * 0.5 * (vsize - 1)).round().clamp(0, vsize - 1).long()
    return yv * vsize + xv

# 这是一个实现随机掩码的辅助函数，我们把它放在这里
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
class EnhancedSparseUNet(nn.Module):
    def __init__(self, channels=1, condition_dim=7, nf=64, time_emb_dim=256, img_size=128, 
                 num_conv_blocks=3, knn_neighbours=3, uno_res=64, uno_mults=(1,2,4,8), 
                 out_channels=None, conv_type="conv", depthwise_sparse=True, kernel_size=7, 
                 backend="torch_dense", optimise_dense=True, blocks_per_level=(2,2,2,2), 
                 attn_res=[16,8], dropout_res=16, dropout=0.1, uno_base_nf=64,cond_drop_prob=0.0,
                 virtual_scale=1  # <--- [新增] 默认1兼容旧代码，实际会传10
                 ):
        super().__init__()
        
        # 基础参数
        self.backend = backend
        self.img_size = img_size
        self.uno_res = uno_res
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size
        self.optimise_dense = optimise_dense
        self.condition_dim = condition_dim
        self.precip_channels = channels
        self.cond_drop_prob = cond_drop_prob # <--- 存储参数
        # 保存这个参数
        self.virtual_scale = virtual_scale
        # 🔥 修改：不再使用可学习参数，直接在forward中创建全0嵌入
        # 原来的代码：self.null_condition_embedding = nn.Parameter(torch.randn(1, 1, condition_dim))
        # 现在不需要这个参数了
        # in __init__
        self.null_condition_embedding = nn.Parameter(torch.zeros(1, 1, condition_dim))
        nn.init.normal_(self.null_condition_embedding, mean=0.0, std=0.02)

        # 输入维度 = 降水 + 条件 (Cat方案)
        input_channels = channels + condition_dim
        #input_channels = channels
        # Input projection (处理cat后的输入)
        self.linear_in = nn.Linear(input_channels, nf)
        
        # Output projection (只输出降水)
        self.linear_out = nn.Linear(nf, out_channels if out_channels is not None else channels)

        # 时间嵌入 (保留)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # UNO坐标
        xs = torch.linspace(-1, 1, steps=uno_res)
        ys = torch.linspace(-1, 1, steps=uno_res)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")  # gx:x, gy:y
        uno_coords = torch.stack([gx, gy], dim=0)       # (x,y)
        uno_coords = rearrange(uno_coords, 'c h w -> () (h w) c')
        self.register_buffer("uno_coords", uno_coords)


        self.normalising_conv = get_normalising_conv(kernel_size=kernel_size, backend=backend)

        # Down blocks (去除z_dim，添加交叉注意力)
        self.down_blocks = nn.ModuleList([])
        self.down_cross_attns = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, 
                time_emb_dim=time_emb_dim, z_dim=None,  # ← 去除z_dim
                depthwise=depthwise_sparse, backend=backend
            ))
            # 每个down block后添加交叉注意力
            #self.down_cross_attns.append(CrossAttentionCondition(nf, condition_dim))

        # UNO相关
        self.uno_linear_in = nn.Linear(nf, uno_base_nf)
        self.uno_linear_out = nn.Linear(uno_base_nf, nf)
        
        # UNO (去除z_dim)
        self.uno = UNO(uno_base_nf, uno_base_nf, width=uno_base_nf, mults=uno_mults, 
                       blocks_per_level=blocks_per_level, time_emb_dim=time_emb_dim, 
                       z_dim=None, conv_type=conv_type, res=uno_res,  # ← 去除z_dim
                       attn_res=attn_res, dropout_res=dropout_res, dropout=dropout)

        # Up blocks (去除z_dim，添加交叉注意力)
        self.up_blocks = nn.ModuleList([])
        self.up_cross_attns = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.up_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, skip_dim=nf, 
                time_emb_dim=time_emb_dim, z_dim=None,  # ← 去除z_dim
                depthwise=depthwise_sparse, backend=backend
            ))
            # 每个up block后添加交叉注意力
            #self.up_cross_attns.append(CrossAttentionCondition(nf, condition_dim))

    def knn_interpolate_to_grid(self, x, coords):
        grid_coords_repeated = self.uno_coords.repeat(x.size(0), 1, 1)
        interpolated_features = knn_interpolate(
            features=x,
            coords=coords,
            grid_coords=grid_coords_repeated,
            k=self.knn_neighbours
        )
        return interpolated_features

    def forward(
        self,
        x,                      # [B,L,C] (sparse) 或 [B,C,H,W] (dense, 自动兼容)
        t,                      # [B]
        condition=None,         # [B,L,D] 或 [B,D,H,W] (自动兼容)
        sample_lst=None,        # [B,L]
        coords=None,            # [B,L,2]
        cond_drop_prob=None,
        **kwargs,               # 兼容 diffusion 可能传的 z 等
    ):
        """
        x:         [B, L, precip_channels] 或 [B, C, H, W]
        condition: [B, L, condition_dim]   或 [B, D, H, W]
        """
        # ==========================================
        # 🔥 1. 入口：自动兼容全图模式 (Dense Mode)
        # ==========================================
        is_dense_input = (x.ndim == 4)  # 判断是否为全图输入
        input_dense_shape = x.shape if is_dense_input else None

        if is_dense_input:
            B, C, H, W = x.shape
            num_points = H * W
            
            # A. 自动生成全图索引 (如果 kwargs 里没传的话)
            if sample_lst is None:
                # [1, H*W] -> [B, H*W]
                sample_lst = torch.arange(num_points, device=x.device).unsqueeze(0).repeat(B, 1)
            
            # B. 把全图 x 变成稀疏格式 [B, H*W, C]
            # (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
            
            # C. 自动生成全图坐标 coords (如果没传的话)
            if coords is None:
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, steps=H, device=x.device),
                    torch.linspace(-1, 1, steps=W, device=x.device),
                    indexing='ij'
                )
                # [H, W, 2] -> [B, H*W, 2]
                grid_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).repeat(B, 1, 1)
                coords = grid_coords

            # D. 处理 Condition (如果 Condition 也是全图)
            if condition is not None and condition.ndim == 4:
                # [B, D, H, W] -> [B, H*W, D]
                condition = condition.permute(0, 2, 3, 1).reshape(B, -1, condition.shape[1]).contiguous()

        # ==========================================
        #      👇 以下是你原来的核心逻辑 (几乎没动) 👇
        # ==========================================
        batch_size, num_points, _ = x.shape
        device = x.device

        assert sample_lst is not None, "sample_lst must be provided."
        assert coords is not None, "coords must be provided (geo coords)."
        assert coords.ndim == 3 and coords.shape[-1] == 2, f"coords must be [B,L,2], got {coords.shape}"

        if condition is not None:
            assert condition.ndim == 3 and condition.shape[:2] == x.shape[:2], \
                f"condition must be [B,L,D], got {condition.shape}, x is {x.shape}"
            assert condition.shape[-1] == self.condition_dim, \
                f"condition last dim mismatch: got {condition.shape[-1]}, expect {self.condition_dim}"

        if x.dtype == torch.float16 and t.dtype == torch.float32:
            t = t.half()

        # ---- CFG condition dropping ----
        drop_prob = self.cond_drop_prob if cond_drop_prob is None else float(cond_drop_prob)
        radar_dim = 5
        if (drop_prob > 0.0) and (condition is not None):
            keep = prob_mask_like((batch_size,), 1.0 - drop_prob, device=device).view(batch_size, 1, 1)
            radar = condition[..., :radar_dim]
            lonlat = condition[..., radar_dim:]
            radar = torch.where(keep.expand_as(radar), radar, torch.zeros_like(radar))
            condition = torch.cat([radar, lonlat], dim=-1)

        # Cat 条件到输入
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)  # [B,L,precip+cond]

        # 输入投影 + 时间嵌入
        x = self.linear_in(x)
        t = self.time_mlp(t)

        # 稀疏处理 (如果 coords 还没生成的话，这里生成逻辑保留作为兜底)
        if coords is None: # 注意：前面如果是全图模式已经生成过了
            coords = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, steps=self.img_size, device=x.device),
                torch.linspace(-1, 1, steps=self.img_size, device=x.device),
                indexing="ij"
            ))
            coords = rearrange(coords, 'c h w -> () (h w) c')
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            coords = torch.gather(coords, 1, sample_lst.unsqueeze(2).repeat(1,1,coords.size(2))).contiguous()

        sample_lst_virt = coords_pix_to_virtual_sample_lst(
                            coords_pix_norm=coords,
                            img_size=self.img_size,
                            virtual_scale=self.virtual_scale
                        )
        
        # ---------------------------------------------------------
        x = convert_to_backend_form(
            x, sample_lst_virt, self.img_size,
            backend=self.backend,
            virtual_scale=self.virtual_scale
        )
        
        backend_tensor = x
        norm = calculate_norm(
            self.normalising_conv,
            backend_tensor,
            sample_lst_virt,
            self.img_size,
            batch_size,
            backend=self.backend
        )
       
        # 1. Down blocks
        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, norm=norm)
            downs.append(x)
            
        # 2. UNO处理
        x = get_features_from_backend_form(x, sample_lst_virt, backend=self.backend)

        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res)

        # UNO
        x = self.uno(x, t)

        # 插值回稀疏坐标
        # F.grid_sample 需要 coords 形状为 [B, H, W, 2]
        # 这里的 coords 是 [B, L, 2]，我们把它unsqueeze成 [B, L, 1, 2]
        x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear', align_corners=True)
        x = rearrange(x, "b c l () -> b l c")
        x = self.uno_linear_out(x)
        x = convert_to_backend_form_like(
            x, backend_tensor,
            sample_lst=sample_lst,
            img_size=self.img_size,
            backend=self.backend
        )

        # 3. Up blocks
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, skip=skip, norm=norm)
            
        # 输出投影
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        output = self.linear_out(x) # [B, L, C_out]

        # ==========================================
        # 🔥 2. 出口：自动还原全图模式
        # ==========================================
        if is_dense_input:
            B, L, C_out = output.shape
            H, W = input_dense_shape[2], input_dense_shape[3]
            
            # [B, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
            output = output.reshape(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()

        return output
        
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