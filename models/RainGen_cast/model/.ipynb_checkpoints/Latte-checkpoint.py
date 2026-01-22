# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# Modified for radar echo extrapolation with variable resolution support.
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

# 省略 Attention, TimestepEmbedder, LabelEmbedder, TransformerBlock, FinalLayer 等不变的部分
# 这里只粘贴核心的 Latte 类和位置编码函数，以及必要的辅助类。

# --- Start of Unchanged Helper Classes (for completeness) ---

class Attention(nn.Module):
    # ... (代码与您提供的一致)
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)

        # Using math-based attention for stability with AMP
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TimestepEmbedder(nn.Module):
    # ... (代码与您提供的一致)
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        return self.mlp(t_freq)

def modulate(x, shift, scale):
    # A helper function for adaLN-Zero
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerBlock(nn.Module):
    # ... (代码与您提供的一致)
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    # ... (代码与您提供的一致)
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# --- End of Unchanged Helper Classes ---


class Latte(nn.Module):
    """
    Diffusion model with a Transformer backbone, modified for radar echo extrapolation.
    This version uses concatenation for conditioning and supports variable resolution inference.
    """
    def __init__(
        self,
        input_size=20,           # Latent space spatial size, e.g., 160 // 8 = 20
        patch_size=2,            # Patch size
        in_channels=64,          # Latent channels from VAE
        hidden_size=768,         # Transformer hidden dimension
        depth=12,                # Number of Transformer blocks
        num_heads=12,            # Number of attention heads
        mlp_ratio=4.0,           # MLP expansion ratio
        num_frames=30,           # Number of frames to predict (output)
        num_condition_frames=10, # Number of condition frames (input)
        learn_sigma=True,        # Whether to learn the variance of the diffusion process
        attention_mode='math',   # Use 'math' for stability with AMP
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_condition_frames = num_condition_frames
        self.total_frames = num_frames + num_condition_frames

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches
        # Positional embeddings are pre-computed but stored in nn.Parameter for convenience
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, self.total_frames, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # 使用标准的 Xavier/Glorot 初始化，而不是全部置零
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by 2D sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize (and freeze) temp_embed by 1D sin-cos embedding:
        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.total_frames)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], "Number of patches must form a square grid."

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def get_dynamic_pos_embed(self, num_patches_current, device):
        """
        Calculates or interpolates positional embeddings for the current input resolution.
        """
        num_patches_precomputed = self.pos_embed.shape[1]
        
        if num_patches_current == num_patches_precomputed:
            return self.pos_embed

        # Interpolate positional embeddings if resolution changes
        grid_size_precomputed = int(num_patches_precomputed ** 0.5)
        pos_embed_reshaped = self.pos_embed.reshape(1, grid_size_precomputed, grid_size_precomputed, -1).permute(0, 3, 1, 2)
        
        grid_size_current = int(num_patches_current ** 0.5)
        pos_embed_interpolated = F.interpolate(
            pos_embed_reshaped,
            size=(grid_size_current, grid_size_current),
            mode='bilinear',
            align_corners=False
        )
        
        return pos_embed_interpolated.permute(0, 2, 3, 1).reshape(1, num_patches_current, -1)

    def forward(self, 
                x,                # Noisy target frames: (B, T_out, C, H, W)
                t,                # Timesteps: (B,)
                condition,        # Condition frames: (B, T_in, C, H, W)
                use_fp16=False):
        """
        Forward pass tailored for video extrapolation.
        """
        # Ensure input tensor format is correct (B, T, C, H, W)
        if x.ndim != 5 or condition.ndim != 5:
            raise ValueError(f"Expected 5D tensors, but got x: {x.ndim}D and condition: {condition.ndim}D")
        
        # 1. Concatenate condition and noisy target frames along the time dimension
        full_seq = torch.cat([condition, x], dim=1)
        
        if use_fp16:
            full_seq = full_seq.to(dtype=torch.float16)

        batches, total_frames_actual, channels, high, weight = full_seq.shape
        assert total_frames_actual == self.total_frames, f"Input frames {total_frames_actual} != expected {self.total_frames}"

        # Reshape for patch embedding: (B * F, C, H, W)
        x_seq = rearrange(full_seq, 'b f c h w -> (b f) c h w')
        x_seq = self.x_embedder(x_seq)

        # Add (interpolated) positional embedding
        pos_embed = self.get_dynamic_pos_embed(x_seq.shape[1], x_seq.device)
        x_seq = x_seq + pos_embed
        
        # Get timestep embedding for modulation
        t_emb = self.t_embedder(t, use_fp16=use_fp16)
        
        # Prepare modulation signals for spatial and temporal blocks
        # Here, only timestep is used for modulation as condition is part of the sequence.
        timestep_spatial = repeat(t_emb, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        timestep_temp = repeat(t_emb, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        # Core Transformer blocks (alternating spatial and temporal attention)
        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            
            x_seq = spatial_block(x_seq, timestep_spatial)
            
            x_seq = rearrange(x_seq, '(b f) t d -> (b t) f d', b=batches)
            if i == 0: # Add temporal embedding only once
                x_seq = x_seq + self.temp_embed
            x_seq = temp_block(x_seq, timestep_temp)
            x_seq = rearrange(x_seq, '(b t) f d -> (b f) t d', b=batches)

        # Final layer
        x_seq = self.final_layer(x_seq, timestep_spatial)
        x_seq = self.unpatchify(x_seq)
        x_seq = rearrange(x_seq, '(b f) c h w -> b f c h w', b=batches)
        
        # 2. Extract only the predicted part of the sequence
        output = x_seq[:, self.num_condition_frames:, ...]
        
        return output

    def forward_with_cond_scale(self, x, t, condition, cond_scale, **kwargs):
        """
        A method to be compatible with your existing GaussianDiffusion framework which uses CFG.
        This method will not be called directly if you handle CFG inside GaussianDiffusion.
        It's provided here as a reference and for full compatibility.
        """
        # CFG requires a conditional and an unconditional pass.
        # The unconditional pass is done by providing a null condition (e.g., zeros).
        
        # First, run the conditional model
        cond_pred = self.forward(x, t, condition)
        
        # Second, run the unconditional model
        null_condition = torch.zeros_like(condition)
        uncond_pred = self.forward(x, t, null_condition)
        
        # Combine them using the CFG formula
        return uncond_pred + cond_scale * (cond_pred - uncond_pred)


# --- Positional Embedding Functions (unchanged) ---
def get_1d_sincos_temp_embed(embed_dim, length):
    # ...
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    # ...
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # ...
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # ...
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# --- Model Factory Functions (for easy instantiation) ---
def Latte_B_2_Radar(**kwargs):
    return Latte(depth=12, hidden_size=768, num_heads=12, **kwargs)

def Latte_L_2_Radar(**kwargs):
    return Latte(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# etc.

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Configuration ---
    B, T_in, T_out = 2, 10, 30
    H, W, C = 160, 160, 4 # Example VAE downsampled size and channels
    patch_size = 2
    latent_H, latent_W = H // 8, W // 8 # 20, 20

    # --- Inputs ---
    condition_frames = torch.randn(B, T_in, C, latent_H, latent_W).to(device)
    noisy_target_frames = torch.randn(B, T_out, C, latent_H, latent_W).to(device)
    timesteps = torch.randint(0, 1000, (B,)).to(device)

    # --- Model Instantiation ---
    model = Latte_B_2_Radar(
        input_size=latent_H,
        patch_size=patch_size,
        in_channels=C,
        num_frames=T_out,
        num_condition_frames=T_in
    ).to(device)

    print("Model instantiated successfully.")
    
    # --- Forward Pass Test ---
    output = model(noisy_target_frames, timesteps, condition_frames)
    print(f"Forward pass successful. Output shape: {output.shape}")
    assert output.shape == noisy_target_frames.shape

    # --- Test Variable Resolution ---
    print("\n--- Testing Variable Resolution ---")
    H_large, W_large = 240, 320
    latent_H_large, latent_W_large = H_large // 8, W_large // 8
    
    # NOTE: For non-square, PatchEmbed needs to handle it. 
    # The current sinusoidal embedding assumes a square grid. Let's test with a larger square.
    latent_H_large, latent_W_large = 32, 32
    condition_large = torch.randn(B, T_in, C, latent_H_large, latent_W_large).to(device)
    noisy_target_large = torch.randn(B, T_out, C, latent_H_large, latent_W_large).to(device)
    
    # Re-instantiate model for the new base resolution if needed (for PatchEmbed)
    # or ensure PatchEmbed is resolution-agnostic.
    # The default timm.PatchEmbed can handle it.
    model_large_res_test = Latte_B_2_Radar(
        input_size=latent_H_large, # PatchEmbed needs to know the base resolution
        patch_size=patch_size,
        in_channels=C,
        num_frames=T_out,
        num_condition_frames=T_in
    ).to(device)
    # In a real scenario, you train on one size, and infer on another.
    # Let's simulate that by using the original model.
    print(f"Inferring on {latent_H_large}x{latent_W_large} with model trained for {latent_H}x{latent_H}...")
    output_large = model(noisy_target_large, timesteps, condition_large)
    print(f"Variable resolution forward pass successful. Output shape: {output_large.shape}")
    assert output_large.shape == noisy_target_large.shape