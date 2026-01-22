# file: train_enhanced_sparse_unet.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops import rearrange
import warnings
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import copy
import shutil
import yaml
import argparse
import math
import glob
# --- 导入你的自定义模块 ---
# 确保这些文件与此脚本在同一个Python模块路径下
from models.model_cfg_new import EnhancedSparseUNet  # ← 新的增强模型
from models.diffusion_new import (
    GaussianDiffusion, get_named_beta_schedule, ModelMeanType,
    ModelVarType, LossType, SpacedDiffusion, space_timesteps, get_conv, DCTGaussianBlur
)
# 使用我们之前讨论过的Z-Score版本的Dataset
from models.dataset import RadarPrecipitationFixedRegionDataset, custom_collate_fn

# --- 环境设置 ---
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
def get_random_vis_sample(val_dataset, H):
    """
    每次验证时随机选择一个可视化样本
    """
    if len(val_dataset) == 0:
        return None, None
    
    # 随机选择样本，最多尝试10次找到有效样本
    for _ in range(10):
        sample_idx = random.randint(0, len(val_dataset) - 1)
        vis_data = val_dataset[sample_idx]
        if vis_data is not None:
            # 计算样本降水信息用于日志
            try:
                precip_patch = vis_data["combined_patch"][:H["data"]["precip_channels"], :, :]
                precip_log1p = (precip_patch * H["data"]["precip_std"]) + H["data"]["precip_mean"]
                precip_physical = torch.expm1(precip_log1p).clamp(min=0)
                
                sample_info = {
                    'index': sample_idx,
                    'mean_precip': precip_physical.mean().item(),
                    'max_precip': precip_physical.max().item(),
                }
                return vis_data, sample_info
            except:
                continue
    
    return None, None

def scatter_points_to_grid(points, indices, grid_shape):
    """
    将稀疏点云数据散布回一个稠密的网格中。

    Args:
        points (torch.Tensor): 稀疏点云数据。
                               形状: [B, num_points, C]
        indices (torch.Tensor): 每个点在展平网格中的索引。
                                形状: [B, num_points]
        grid_shape (tuple): 目标网格的形状。
                            格式: (B, C, H, W)

    Returns:
        torch.Tensor: 包含了散布后数据的稠密网格。
                      形状: [B, C, H, W]
    """
    B, C, H, W = grid_shape
    num_points = indices.shape[1]
    
    # 1. 创建一个空白的目标网格（画布）
    grid = torch.zeros(grid_shape, device=points.device, dtype=points.dtype)
    
    # 2. 将画布展平，以便使用 scatter_
    # 形状: [B, H*W, C]
    flat_grid = rearrange(grid, 'b c h w -> b (h w) c')
    
    # 3. 准备索引张量，使其维度与 points 兼容
    # scatter_ 要求 index 和 src(points) 的维度数相同
    # 形状从 [B, num_points] -> [B, num_points, 1]
    # 然后扩展到 [B, num_points, C]
    indices_expanded = indices.unsqueeze(2).expand(-1, -1, C)

    # 4. 执行散布操作（原地修改 flat_grid）
    # 将 points 中的值，根据 indices_expanded 的指示，写入到 flat_grid 中
    flat_grid.scatter_(dim=1, index=indices_expanded, src=points)
    
    # 5. 将填充好的展平网格重新塑形回图像格式
    # 注意：这里我们直接返回重塑后的 flat_grid，而不是原始的 grid
    final_grid = rearrange(flat_grid, 'b (h w) c -> b c h w', h=H, w=W)
    
    return final_grid
# ==============================================================================
# 辅助函数
# ==============================================================================
def _gather_from_grid(grid, sample_lst):
    """
    与Diffusion中的实现保持一致
    """
    grid = rearrange(grid, 'b c h w -> b (h w) c')
    indices = sample_lst.unsqueeze(2).repeat(1, 1, grid.size(2))
    points = torch.gather(grid, 1, indices).contiguous()
    return points

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return int(os.environ["LOCAL_RANK"])
    else:
        print("INFO: Not in a DDP environment. Running in single-process mode.")
        return 0

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()

def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        model_for_ema = model.module if hasattr(model, 'module') else model
        for ema_param, param in zip(ema_model.parameters(), model_for_ema.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def save_checkpoint(state, is_best, output_dir, filename="checkpoint.pth.tar"):
    if not is_main_process(): return
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(output_dir, 'model_best.pth.tar'))
        print(f"🏆 Best model saved to {os.path.join(output_dir, 'model_best.pth.tar')}")
    else:
        print(f"✓ Checkpoint saved to {filepath}")

def load_checkpoint(denoiser, ema_denoiser, optimizer, scheduler, output_dir, device):
    """修改：移除encoder相关加载"""
    filepath = os.path.join(output_dir, "checkpoint.pth.tar")
    start_epoch, global_step, best_val_loss = 0, 0, float('inf')
    if os.path.exists(filepath):
        if is_main_process(): print(f"📂 Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        try:
            (denoiser.module if hasattr(denoiser, 'module') else denoiser).load_state_dict(checkpoint['denoiser_state_dict'])
            ema_denoiser.load_state_dict(checkpoint['ema_denoiser_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if is_main_process(): print(f"✓ Resumed from epoch {start_epoch}, global step {global_step}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}. Starting fresh.")
            start_epoch, global_step, best_val_loss = 0, 0, float('inf')
    return start_epoch, global_step, best_val_loss

# ==============================================================================
# 验证与可视化
# ==============================================================================
# train_enhanced_sparse_unet.py

@torch.no_grad()
def validate(diffusion, denoiser_ema, val_loader, H, device, global_step):
    denoiser_ema.eval()
    total_loss = 0.0
    num_val_batches = 0
    pbar_val = tqdm(val_loader, desc="Validating", leave=False, disable=not is_main_process())
    
    for batch_data in pbar_val:
        if batch_data is None: continue
        
        combined_patch = batch_data["combined_patch"].to(device, non_blocking=True)
        coords_abs_grid = batch_data["coords_abs_patch"].to(device, non_blocking=True)
        coords_geo_grid = batch_data["coords_geo_patch"].to(device, non_blocking=True)

        precip_grid = combined_patch[:, :H["data"]["precip_channels"], ...]
        radar_grid  = combined_patch[:, H["data"]["precip_channels"]:, ...]

        B, _, H_crop, W_crop = precip_grid.shape
        num_points = min(H["mc_integral"]["q_sample"], H_crop * W_crop)
        
        # 1. 生成随机索引
        sample_lst = torch.stack([
            torch.from_numpy(np.random.choice(H_crop * W_crop, num_points, replace=False))
            for _ in range(B)
        ]).to(device)

        # 🔥🔥🔥【关键修改】🔥🔥🔥
        # 必须对索引进行排序！SpConv 内部是按空间顺序处理的。
        # 如果这里不排序，gather 出来的特征是乱序的，但 SpConv 输出是排序的，导致错位。
        sample_lst, _ = torch.sort(sample_lst, dim=1) 

        # 2. 使用【排序后】的 sample_lst 进行 gather
        # 这样 input(radar) 和 target(precip) 都是按照空间顺序排列的 (row-major)
        radar_points = _gather_from_grid(radar_grid, sample_lst)            # [B,L,5]
        coords_abs_points = _gather_from_grid(coords_abs_grid, sample_lst)  # [B,L,2]
        coords_geo_points = _gather_from_grid(coords_geo_grid, sample_lst)  # [B,L,2]

        condition = torch.cat([radar_points, coords_abs_points], dim=-1)    # [B,L,7]

        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
        model_kwargs = {
            "condition": condition,
            "sample_lst": sample_lst, # 传入排序后的索引
            "coords": coords_geo_points
        }

        # Loss 计算时，diffusion 内部会用 sample_lst 去 gather x_start (precip)
        # 因为 sample_lst 已经排序，所以取出的 precip 也是有序的，与模型输出对应。
        loss_dict = diffusion.training_losses(
            model=denoiser_ema, 
            x_start=precip_grid, 
            t=t,
            sample_lst=sample_lst, 
            model_kwargs=model_kwargs
        )
        loss = loss_dict["loss"].mean()

        if not torch.isnan(loss):
            total_loss += loss.item()
        
        num_val_batches += 1
            
    avg_loss = total_loss / num_val_batches if num_val_batches > 0 else float('inf')
    return avg_loss

# file: train_Diffusion.py

@torch.no_grad()
def visualize_and_log(diffusion, denoiser_ema, vis_data, epoch, global_step, H, output_dir):
    if not is_main_process() or vis_data is None: return
    print(f"\n🎨 Generating visualization for step {global_step}...")
    denoiser_ema.eval()
    device = next(denoiser_ema.parameters()).device
    
    # 1. 准备数据 (全图)
    # [1, C_total, H, W]
    combined_patch = vis_data["combined_patch"].to(device).unsqueeze(0)
    coords_abs_grid = vis_data["coords_abs_patch"].to(device).unsqueeze(0)
    coords_geo_grid = vis_data["coords_geo_patch"].to(device).unsqueeze(0)

    # 拆分降水(Target) 和 雷达(Condition)
    precip_grid = combined_patch[:, :H["data"]["precip_channels"]]
    radar_grid  = combined_patch[:, H["data"]["precip_channels"]:]
    
    B, _, H_crop, W_crop = precip_grid.shape
    
    # 2. 准备 Condition (保持全图形状，不 Gather！)
    # [1, 5, H, W] + [1, 2, H, W] -> [1, 7, H, W]
    condition_dense = torch.cat([radar_grid, coords_abs_grid], dim=1)
    
    # 准备坐标 (虽然模型能自动处理，但为了保险，展平传进去)
    # [1, 2, H, W] -> [1, H*W, 2]
    coords_geo_flat = rearrange(coords_geo_grid, 'b c h w -> b (h w) c')

    # 构造传给模型的参数
    # 注意：我们不需要传 'sample_lst'，因为模型检测到全图输入会自动生成全图索引
    model_kwargs = {
        "condition": condition_dense,  # 🔥 直接传全图，模型Forward里会自动处理
        "coords": coords_geo_flat      # 坐标
    }

    # 3. 初始化推理用的 Diffusion (开启 DCT)
    betas = get_named_beta_schedule(H["diffusion"]["noise_schedule"], H["diffusion"]["steps"])
    ddim_diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion.num_timesteps, f"ddim{H['inference'].get('ddim_steps', 50)}"),
        betas=betas,
        model_mean_type=ModelMeanType.MOLLIFIED_EPSILON, # 必须一致
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        precip_channels=H["data"]["precip_channels"],
        rescale_timesteps=True,
        mollifier_type="dct",            # ✅ 必须开启 DCT
        img_size=H["data"]["crop_size"]  # ✅ 必须填
    )

    try:
        # 🔥 4. 采样
        # shape 必须是全图形状，这样 DCT 才能工作
        dense_shape = (B, H["data"]["precip_channels"], H_crop, W_crop)
        
        samples_grid, _ = ddim_diffusion.ddim_sample_loop(
            model=denoiser_ema, # 🔥 直接传模型，不需要 Wrapper 了！
            shape=dense_shape,  # 🔥 告诉它是全图
            model_kwargs=model_kwargs,
            progress=True,
            eta=H["inference"].get("eta", 0.0),
            guidance_scale=H["inference"].get("guidance_scale", 3.0)
        )
        print("✅ DDIM sampling successful!")
        
        # --- 5. 绘图 (和之前一样) ---
        vis_path = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_path, exist_ok=True)
        
        # 反归一化
        pred_vis = (samples_grid[0, 0].cpu().numpy() * H["data"]["precip_std"]) + H["data"]["precip_mean"]
        true_vis = (precip_grid[0, 0].cpu().numpy() * H["data"]["precip_std"]) + H["data"]["precip_mean"]
        radar_vis = (radar_grid[0, 0].cpu().numpy() * H["data"]["radar_std"]) + H["data"]["radar_mean"]
        
        # Clip negative values
        pred_vis = np.maximum(pred_vis, 0)
        true_vis = np.maximum(true_vis, 0)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(radar_vis, cmap='jet'); ax[0].set_title("Input Radar")
        ax[1].imshow(true_vis, cmap='Blues', vmin=0, vmax=pred_vis.max()); ax[1].set_title("Ground Truth")
        ax[2].imshow(pred_vis, cmap='Blues', vmin=0, vmax=pred_vis.max()); ax[2].set_title(f"Generated (Step {global_step})")
        
        save_file = os.path.join(vis_path, f"epoch_{epoch}_step_{global_step}.png")
        plt.savefig(save_file)
        plt.close()

    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback; traceback.print_exc()
# ==============================================================================
# 主函数
# ==============================================================================
def main(H):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    output_dir = H["run"]["output_dir"]
    
    writer = None
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(H, f, default_flow_style=False)
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        print(f"📁 Output directory: {output_dir}")

    # 新的代码：使用patch数据目录
    patch_data_dir = H["data"]["patch_data_dir"]  # 在config中新增这个字段
    all_files = sorted(glob.glob(os.path.join(patch_data_dir, "*.npz")))
    
    if not all_files: 
        raise ValueError(f"No .npz patch files found in {patch_data_dir}")
    
    print(f"📊 Found {len(all_files)} patch files in {patch_data_dir}")
    
    random.seed(42); random.shuffle(all_files)
    train_files, val_files = np.split(np.array(all_files), [int(len(all_files) * 0.9)])
    
    # 使用新的PatchedDataset
    train_dataset = RadarPrecipitationFixedRegionDataset(
        file_list=train_files.tolist(), 
        normalize=H["data"]["normalize"],
        radar_mean=H["data"]["radar_mean"],
        radar_std=H["data"]["radar_std"],
        precip_mean=H["data"]["precip_mean"],
        precip_std=H["data"]["precip_std"],
        crop_size=H["data"]['crop_size']
    )
    
    val_dataset = RadarPrecipitationFixedRegionDataset(
        file_list=val_files.tolist(),
        normalize=H["data"]["normalize"],
        radar_mean=H["data"]["radar_mean"],
        radar_std=H["data"]["radar_std"],
        precip_mean=H["data"]["precip_mean"],
        precip_std=H["data"]["precip_std"],
        crop_size=H["data"]['crop_size']
    )
    
    # === 🔥 修改结束 ===
    
    # DataLoader部分保持不变
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    
    train_loader = DataLoader(train_dataset, batch_size=H["train"]["batch_size_per_gpu"], sampler=train_sampler, shuffle=(train_sampler is None), num_workers=H["train"]["num_workers"], pin_memory=True, collate_fn=custom_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=H["train"]["batch_size_per_gpu"], sampler=val_sampler, shuffle=False, num_workers=H["train"]["num_workers"], pin_memory=True, collate_fn=custom_collate_fn)


   
    
    print("🔧 Setting up Enhanced Sparse UNet...")
    denoiser = EnhancedSparseUNet(
        channels=H["data"]["precip_channels"],  # 只有降水通道
        condition_dim=H["data"]["radar_channels"] ,  # 5雷达 + 2坐标
        out_channels=H["data"]["precip_channels"],
        nf=H["model"]["nf"],
        time_emb_dim=H["model"]["time_emb_dim"],
        img_size=H["data"]["crop_size"],
        num_conv_blocks=H["model"]["num_conv_blocks"],
        knn_neighbours=H["model"]["knn_neighbours"],
        uno_res=H["model"]["uno_res"],
        uno_mults=tuple(H["model"]["uno_mults"]),
        conv_type=H["model"]["conv_type"],
        depthwise_sparse=H["model"].get("depthwise_sparse", True),
        kernel_size=H["model"].get("kernel_size", 7),
        backend=H["model"].get("backend", "torch_dense"),
        blocks_per_level=(2,2,2,2),
        attn_res=H["model"].get("attn_res", []),
        dropout_res=H["model"].get("dropout_res", 16),
        dropout=H["model"].get("dropout", 0.1),
        uno_base_nf=H["model"].get("uno_base_nf", 64),
        cond_drop_prob=H['model'].get("cond_drop_prob",0.1),
        virtual_scale=H["model"].get("virtual_scale", 1)

    ).to(device)
    
    ema_denoiser = EnhancedSparseUNet(
        channels=H["data"]["precip_channels"],
        condition_dim=H["data"]["radar_channels"],
        out_channels=H["data"]["precip_channels"],
        nf=H["model"]["nf"],
        time_emb_dim=H["model"]["time_emb_dim"],
        img_size=H["data"]["crop_size"],
        num_conv_blocks=H["model"]["num_conv_blocks"],
        knn_neighbours=H["model"]["knn_neighbours"],
        uno_res=H["model"]["uno_res"],
        uno_mults=tuple(H["model"]["uno_mults"]),
        conv_type=H["model"]["conv_type"],
        depthwise_sparse=H["model"].get("depthwise_sparse", True),
        kernel_size=H["model"].get("kernel_size", 7),
        backend=H["model"].get("backend", "torch_dense"),
        blocks_per_level=(2,2,2,2),
        attn_res=H["model"].get("attn_res", []),
        dropout_res=H["model"].get("dropout_res", 16),
        dropout=H["model"].get("dropout", 0.1),
        uno_base_nf=H["model"].get("uno_base_nf", 64),
        cond_drop_prob=H['model'].get("cond_drop_prob",0.1),
        virtual_scale=H["model"].get("virtual_scale", 1)

    ).to(device)

    # 🔥 优化器：只优化denoiser
    optimizer = AdamW(denoiser.parameters(), lr=H["train"]["lr"], betas=(0.9, 0.99))
    
    betas = get_named_beta_schedule(H["diffusion"]["noise_schedule"], H["diffusion"]["steps"])
    
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.MOLLIFIED_EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        precip_channels=1,
        rescale_timesteps=True,
        mollifier_type="dct"
    ).to(device)

    num_training_steps = len(train_loader) * H["train"]["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * H["train"]["warmup_ratio"]), num_training_steps=num_training_steps)
    
    if dist.is_initialized():
        denoiser = DDP(denoiser, device_ids=[local_rank])

    print(f"🎯 Denoiser params: {sum(p.numel() for p in denoiser.parameters()):,}")
    
    start_epoch, global_step, best_val_loss = load_checkpoint(denoiser, ema_denoiser, optimizer, scheduler, output_dir, device)

    print("🚀 Starting Enhanced Sparse UNet training...")
    for epoch in range(start_epoch, H["train"]["epochs"]):
        if train_sampler: train_sampler.set_epoch(epoch)
        denoiser.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{H['train']['epochs']}", disable=not is_main_process())
        for batch_data in pbar:
            if batch_data is None: continue
            
            optimizer.zero_grad()
            
            combined_patch = batch_data["combined_patch"].to(device, non_blocking=True)
            coords_abs_grid = batch_data["coords_abs_patch"].to(device, non_blocking=True)
            coords_geo_grid = batch_data["coords_geo_patch"].to(device, non_blocking=True)

            precip_grid = combined_patch[:, :H["data"]["precip_channels"], ...]
            radar_grid  = combined_patch[:, H["data"]["precip_channels"]:, ...]

            B, _, H_crop, W_crop = precip_grid.shape
            num_points = min(H["mc_integral"]["q_sample"], H_crop * W_crop)
            
            # 1. 生成随机索引
            sample_lst = torch.stack([
                torch.from_numpy(np.random.choice(H_crop * W_crop, num_points, replace=False))
                for _ in range(B)
            ]).to(device)

            # 🔥🔥🔥【关键修改】🔥🔥🔥
            # 对 sample_lst 进行排序
            sample_lst, _ = torch.sort(sample_lst, dim=1)

            # 2. 使用排序后的 sample_lst 提取数据
            radar_points      = _gather_from_grid(radar_grid, sample_lst)          # [B,L,5]
            coords_abs_points = _gather_from_grid(coords_abs_grid, sample_lst)     # [B,L,2]
            coords_geo_points = _gather_from_grid(coords_geo_grid, sample_lst)     # [B,L,2]

            condition = torch.cat([radar_points, coords_abs_points], dim=-1)       # [B,L,7]

            model_kwargs = {
                "condition": condition,
                "sample_lst": sample_lst,
                "coords": coords_geo_points,
            }

            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
            
            denoiser_fn = denoiser.module if hasattr(denoiser, 'module') else denoiser
            
            loss_dict = diffusion.training_losses(
                model=denoiser_fn, 
                x_start=precip_grid, 
                t=t,
                sample_lst=sample_lst, # 传入排序后的列表
                model_kwargs=model_kwargs
            )
            loss = loss_dict["loss"].mean()
            
            # 🔥 只有扩散损失，没有KLD
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), H["train"]["gradient_clip_val"])
                optimizer.step()
                scheduler.step()
                update_ema(ema_denoiser, denoiser, decay=H["train"]["ema_decay"])
            
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                if writer:
                    writer.add_scalar("Loss/diffusion", loss.item(), global_step)
                    writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        if is_main_process():
            avg_val_loss = validate(diffusion, ema_denoiser, val_loader, H, device, global_step)
            print(f"📉 Epoch {epoch+1} | Val Loss: {avg_val_loss:.5f}")
            if writer:
                writer.add_scalar("Val/loss", avg_val_loss, epoch + 1)

            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"🏆 New best model found!")

            save_checkpoint({
                'epoch': epoch + 1, 
                'global_step': global_step,
                'denoiser_state_dict': (denoiser.module if hasattr(denoiser, 'module') else denoiser).state_dict(),
                'ema_denoiser_state_dict': ema_denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss, 
                'config': H
            }, is_best=is_best, output_dir=output_dir)
            
            # 🔥 修改：每次都随机选择新的可视化样本
            if (epoch) % H["train"]["visualize_every_epochs"] == 0:
                print(f"🎲 Selecting random sample for visualization...")
                random_vis_data, sample_info = get_random_vis_sample(val_dataset, H)
                
                if random_vis_data is not None and sample_info is not None:
                    print(f"   Selected sample {sample_info['index']}: "
                          f"mean={sample_info['mean_precip']:.3f} mm/h, "
                          f"max={sample_info['max_precip']:.3f} mm/h")
                    
                    visualize_and_log(diffusion, ema_denoiser, random_vis_data, 
                                    epoch, global_step, H, output_dir)
                else:
                    print("   ⚠️ Could not find valid sample for visualization")

    if writer: writer.close()
    cleanup_ddp()
    print("🎉 Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/tangxiao_iap/INR/config_new.yaml", help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        H = yaml.safe_load(f)
    
    print("✅ Configuration loaded successfully from:", args.config)
    main(H)