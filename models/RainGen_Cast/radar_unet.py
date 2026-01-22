# models/radar_unet_diffusion.py

import torch
import torch.nn as nn
from einops import rearrange

# --- 导入你已经移植过来的 SVD VideoUNet ---
from model.video_model.video_model import VideoUNet
from model.video_model.openaimodel import timestep_embedding # 导入标准的离散时间嵌入函数

# --- 导入你的 vae_utils.py 中的 prob_mask_like 函数 ---
from model.utils_diffusion import prob_mask_like

class SVDUNetBackbone(nn.Module):
    """
    一个完备的、封装了 SVD VideoUNet 的适配器类，旨在作为 Latte 模型的直接替代品。
    - 实现了与 Latte 完全一致的 forward 接口，支持 CFG。
    - 内部处理了 SVD UNet 对 `context` 张量的硬性要求。
    - 设计为与 GaussianDiffusion 框架无缝协作。
    """
    def __init__(
        self,
        # --- 任务特定参数 (与 Latte 对应) ---
        num_condition_frames: int,
        num_frames: int, # 对应 Latte 的 num_frames (即 T_out)
        
        # --- CFG 参数 (与 Latte 对应) ---
        cond_drop_prob: float,
        
        # --- VAE Latent 形状 ---
        input_size: int, # 对应 latent 的 H 和 W
        
        # --- 传递给内部 VideoUNet 的核心参数 ---
        in_channels: int,        # Latent channels, e.g., 64
        model_channels: int,     # UNet 基础通道数, e.g., 320
        out_channels: int,       # 输出通道数, 通常等于 in_channels
        context_dim= None,        # 🔥 关键：必须从config接收，即使我们不用它
        **kwargs                 # 接收并透传所有其他 VideoUNet 参数
    ):
        super().__init__()
        print("✅ Initializing SVDUNetBackbone Wrapper...")
        self.num_condition_frames = num_condition_frames
        self.num_target_frames = num_frames
        self.total_frames = num_condition_frames + num_frames
        self.cond_drop_prob = cond_drop_prob
        # ↓↓↓ 【新增】将 context_dim 保存为类的属性 ↓↓↓
        #self.context_dim = context_dim
        # 1. 实例化 SVD 的 VideoUNet 作为内部引擎
        self.unet = VideoUNet(
            in_channels=in_channels,
            out_channels=out_channels, 
            model_channels=model_channels,
            
            **kwargs
        )
        print("   - Internal SVD VideoUNet instantiated successfully.")

        # 2. 创建可学习的空条件嵌入 (与 Latte 逻辑一致)
        #null_condition_shape = (1, num_condition_frames, in_channels, input_size, input_size)
        ##self.null_condition_embedding = nn.Parameter(torch.randn(null_condition_shape) * 0.02)
        #print(f"   - Learnable null condition created with shape: {null_condition_shape}")
        # 2. 添加下面这行
        
        null_condition_shape = (1, num_condition_frames, in_channels, input_size, input_size)
        self.register_buffer('null_condition_embedding', torch.zeros(null_condition_shape))
        #self.null_condition_embedding = nn.Parameter(torch.zeros(null_condition_shape))
        
        # 打印一下确认参数是可训练的
        #print(f"   - Learnable Null Condition created: shape={null_condition_shape}, requires_grad={self.null_condition_embedding.requires_grad}"

    def get_null_condition(self, batch_size, device):
        """获取空条件嵌入，扩展到指定批量大小"""
        # nn.Parameter 会自动随模型移动到 GPU，这里 .to(device) 是双重保险
        return self.null_condition_embedding.expand(batch_size, -1, -1, -1, -1).to(device)

    def forward(
        self, 
        x: torch.Tensor,                # 加噪的目标 latent: (B, T_out, C, H, W)
        t: torch.Tensor,                # 离散整数时间步: (B,)
        condition: torch.Tensor,        # 条件 latent: (B, T_in, C, H, W)
        apply_cfg_dropout: bool = True, # 与Latte接口对齐，用于训练
        condition_mask: torch.Tensor = None # 与Latte接口对齐，用于推理
    ):
        b, t_out, c, h, w = x.shape
        device = x.device
        
        # --- 1. 与 Latte 完全一致的 CFG 逻辑 ---
        if apply_cfg_dropout :
            # 训练时：随机条件丢弃
            keep_mask = prob_mask_like((b,), 1 - self.cond_drop_prob, device=device)
            null_condition = self.get_null_condition(b, device)
            mask_shape = [b] + [1] * (condition.ndim - 1)
            condition = torch.where(keep_mask.reshape(mask_shape), condition, null_condition)
        elif condition_mask is not None :
            # 推理时：根据显式 mask 决定条件使用
            null_condition = self.get_null_condition(b, device)
            mask_shape = [b] + [1] * (condition.ndim - 1)
            condition = torch.where(condition_mask.reshape(mask_shape), condition, null_condition)
        '''
        if apply_cfg_dropout:
            # 训练模式：随机丢弃条件
            # 这里的 1 - self.cond_drop_prob 是"保留概率"
            keep_mask = prob_mask_like((b,), 1 - self.cond_drop_prob, device=device)
            condition = torch.where(keep_mask.reshape(mask_shape), condition, null_condition)
            
        elif condition_mask is not None:
            # 推理模式：手动指定哪些样本使用条件，哪些使用空条件
            # condition_mask 为 True 使用 condition，为 False 使用 null_condition
            condition = torch.where(condition_mask.reshape(mask_shape), condition, null_condition)
        '''
        # --- 2. 准备 UNet 输入 ---
        full_sequence = torch.cat([condition.to(x.dtype), x], dim=1)
        unet_input = rearrange(full_sequence, 'b t c h w -> (b t) c h w')
        timesteps_expanded = t.repeat_interleave(self.total_frames)
        image_only_indicator = torch.zeros(b, self.total_frames, device=device)
        
        # --- 3. 🔥 关键修正：创建并传入虚拟 context ---
        # 从 UNet 内部获取其期望的 context_dim
        

        # --- 4. 调用 SVD UNet ---
        predicted_output_flat = self.unet(
            x=unet_input,
            timesteps=timesteps_expanded,
            #context=context_expanded, # <-- 传入虚拟 context
            num_video_frames=self.total_frames,
            image_only_indicator=image_only_indicator
        )
        
        # --- 5. 处理输出 ---
        output_sequence = rearrange(predicted_output_flat, '(b t) c h w -> b t c h w', b=b)
        output_for_target = output_sequence[:, self.num_condition_frames:, ...]
        
        return output_for_target