import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
#from pytorch3d.ops import knn_points, knn_gather
from .knn import *
import math
import warnings

from .conv_uno import UNO, UNOEncoder
from .sparse_conv_block import SparseConvResBlock
from .sparse_conv_block import convert_to_backend_form, convert_to_backend_form_like, \
    calculate_norm, get_features_from_backend_form, get_normalising_conv

# 这是一个实现随机掩码的辅助函数，我们把它放在这里
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

class CrossAttentionCondition(nn.Module):
    def __init__(self, feat_dim, condition_dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        """
        增强版的交叉注意力条件模块
        - 使用MLP深化条件投影
        - 对Query进行LayerNorm
        
        Args:
            feat_dim (int): 特征维度 (Query的维度)
            condition_dim (int): 条件维度 (原始条件的维度)
            num_heads (int): 注意力头数
            mlp_ratio (int): MLP隐藏层维度的放大比例
            dropout (float): Dropout比例
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.condition_dim = condition_dim
        
        # 🔥 改进1: 将条件投影深化为MLP
        # MLP的隐藏层维度通常是输入/输出维度的2倍或4倍
        hidden_dim = feat_dim * mlp_ratio
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        
        # 交叉注意力模块 (保持不变)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # 🔥 改进2: 为Query和输出分别创建LayerNorm
        # 为Query创建一个专门的LayerNorm
        self.query_norm = nn.LayerNorm(feat_dim)
        # 为最终输出的残差连接创建一个LayerNorm
        self.output_norm = nn.LayerNorm(feat_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, condition):
        """
        features: [B, num_points, feat_dim] (作为Query)
        condition: [B, num_points, condition_dim]
        """
        # 1. 投影条件 (现在通过MLP)
        # cond_proj 形状: [B, num_points, feat_dim]
        cond_proj = self.condition_proj(condition)
        
        # 🔥 2. 对Query进行归一化 (使用 query_norm)
        # features_norm 形状: [B, num_points, feat_dim]
        features_norm = self.query_norm(features)
        
        # 3. 交叉注意力计算
        # 使用归一化的 features_norm 作为Query
        attn_out, _ = self.cross_attn(
            query=features_norm,
            key=cond_proj,
            value=cond_proj
        )
        
        # 4. 残差连接 + Dropout + 归一化 (使用 output_norm)
        # 残差连接作用在原始的、未归一化的 features 上
        output = self.output_norm(features + self.dropout(attn_out))
        
        return output

class EnhancedSparseUNet(nn.Module):
    def __init__(self, channels=1, condition_dim=7, nf=64, time_emb_dim=256, img_size=128, 
                 num_conv_blocks=3, knn_neighbours=3, uno_res=64, uno_mults=(1,2,4,8), 
                 out_channels=None, conv_type="conv", depthwise_sparse=True, kernel_size=7, 
                 backend="torch_dense", optimise_dense=True, blocks_per_level=(2,2,2,2), 
                 attn_res=[16,8], dropout_res=16, dropout=0.1, uno_base_nf=64,cond_drop_prob=0.0):
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
        # 🔥 2. 定义可学习的“无条件”嵌入
        self.null_condition_embedding = nn.Parameter(torch.randn(1, 1, condition_dim))
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
        uno_coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=uno_res) for _ in range(2)]))
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
            self.down_cross_attns.append(CrossAttentionCondition(nf, condition_dim))

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
            self.up_cross_attns.append(CrossAttentionCondition(nf, condition_dim))

    def knn_interpolate_to_grid(self, x, coords):
        grid_coords_repeated = self.uno_coords.repeat(x.size(0), 1, 1)
        interpolated_features = knn_interpolate(
            features=x,
            coords=coords,
            grid_coords=grid_coords_repeated,
            k=self.knn_neighbours
        )
        return interpolated_features

    def forward(self, x, t, condition=None, sample_lst=None, coords=None,cond_drop_prob=None):
        """
        x: [B, num_points, precip_channels] 降水特征
        condition: [B, num_points, condition_dim] 条件特征 (5雷达+2坐标)
        """
        batch_size, num_points, _ = x.shape
        device = x.device
        if x.dtype == torch.float16 and t.dtype == torch.float32:
            t = t.half()
        # 🔥 3. 内置的CFG训练逻辑
        drop_prob = cond_drop_prob if cond_drop_prob is not None else self.cond_drop_prob
        if drop_prob > 0. and self.training:
            keep_mask = prob_mask_like((batch_size,), 1. - drop_prob, device=device).view(batch_size, 1, 1)
            null_cond = self.null_condition_embedding
            condition = torch.where(keep_mask, condition, null_cond)
        # 🔥 方案1：直接Cat条件到输入
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)  # [B, num_points, precip_channels + condition_dim]
      
        # 输入投影
        x = self.linear_in(x)
        t = self.time_mlp(t)

        # 稀疏处理
        if coords is None:
            coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=self.img_size) for _ in range(2)])).to(x.device)
            coords = rearrange(coords, 'c h w -> () (h w) c')
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            coords = torch.gather(coords, 1, sample_lst.unsqueeze(2).repeat(1,1,coords.size(2))).contiguous()

        # 转换为稀疏格式
        x = convert_to_backend_form(x, sample_lst, self.img_size, backend=self.backend)
        backend_tensor = x
        norm = calculate_norm(self.normalising_conv, backend_tensor, sample_lst, self.img_size, batch_size, backend=self.backend)

        # 1. Down blocks + 交叉注意力
        downs = []
        for i, (block, cross_attn) in enumerate(zip(self.down_blocks, self.down_cross_attns)):
            # 稀疏卷积块 (只传递时间，不传递z)
            x = block(x, t=t, norm=norm)  # ← 去除z参数
            
            # 🔥 方案2：交叉注意力增强
            if condition is not None:
                # 获取特征进行交叉注意力
                features = get_features_from_backend_form(x, sample_lst, backend=self.backend)
                features = cross_attn(features, condition)
                x = convert_to_backend_form_like(features, x, sample_lst=sample_lst, 
                                               img_size=self.img_size, backend=self.backend)
            
            downs.append(x)

        # 2. UNO处理
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res)

        # UNO (去除z参数)
        x = self.uno(x, t)  # ← 去除z参数

        # 插值回稀疏坐标
        x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear')
        x = rearrange(x, "b c l () -> b l c")
        x = self.uno_linear_out(x)
        x = convert_to_backend_form_like(x, backend_tensor, sample_lst=sample_lst, 
                                       img_size=self.img_size, backend=self.backend)

        # 3. Up blocks + 交叉注意力
        for i, (block, cross_attn) in enumerate(zip(self.up_blocks, self.up_cross_attns)):
            skip = downs.pop()
            
            # 稀疏卷积块 (只传递时间，不传递z)
            x = block(x, t=t, skip=skip, norm=norm)  # ← 去除z参数
            
            # 🔥 方案2：交叉注意力增强
            if condition is not None:
                features = get_features_from_backend_form(x, sample_lst, backend=self.backend)
                features = cross_attn(features, condition)
                x = convert_to_backend_form_like(features, x, sample_lst=sample_lst, 
                                               img_size=self.img_size, backend=self.backend)

        # 输出投影
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.linear_out(x)

        return x
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