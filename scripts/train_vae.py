# train_vae_multicard_working.py - 基于工作单卡版本的多卡适配

import argparse
from torch.cuda.amp import autocast, GradScaler
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from collections import OrderedDict
import json
from torch.utils.tensorboard import SummaryWriter
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed

# 使用与单卡版本完全相同的导入
from vae_utils import *
from models.autoencoder_kl import AutoencoderKL

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='PyTorch VAE Training (Multi-GPU with Accelerate)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers')
parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
parser.add_argument('--resume', default=None, type=str, help='path to checkpoint')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--config', default='/home/tangxiao_iap/DiT/AEKL/config.json', type=str, help='path to config file')
parser.add_argument('--no-amp', action='store_true', help='Disable Automatic Mixed Precision training')

best_loss_val = float("inf")

# --- 使用与单卡版本完全相同的损失函数 ---
def loss_function(recons, input, posterior, kl_weight=1e-6):
    recons_loss = torch.nn.functional.mse_loss(recons, input, reduction='mean')
    kl_loss = posterior.kl().mean()
    total_loss = recons_loss + kl_weight * kl_loss
    return total_loss, recons_loss, kl_loss

# --- 绘图函数 (添加多进程安全) ---
def Plot_radar(image_truth, image_recon, epoch, accelerator, save_dir="/home/tangxiao_iap/DiT/AEKL/logs_latent_64/figs/evaluation"):
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        levs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
        cols = ["#D3D3D3","#01a0f6", "#00ecec", "#6dfa3d", "#00D806", "#019000", "#FFFF00", "#e7c000", "#FF9000", "#FF0000", "#d60000", "#C00000", "#e4007e", "#9600b4", "#AD90F0"]
        cmap = ListedColormap(cols, N=15)
        mae_recon = np.mean(np.abs(image_truth - image_recon))
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        im1 = ax[0].imshow(image_truth, cmap=cmap, vmin=0.0, vmax=70.0)
        ax[0].set_title("Ground Truth", fontsize=14)
        ax[0].axis('off')
        im2 = ax[1].imshow(image_recon, cmap=cmap, vmin=0.0, vmax=70.0)
        ax[1].set_title(f"Reconstruction (MAE={mae_recon:.2f})", fontsize=14)
        ax[1].axis('off')
        cbar = fig.colorbar(im1, ax=ax, orientation='horizontal', shrink=0.8, pad=0.08)
        cbar.set_label('Radar Reflectivity (dBZ)', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"reconstruction_epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"重建图像已保存: {save_path}")

# --- 保存 Checkpoint 函数 (多进程安全版本) ---
def save_checkpoint(state, is_best, epoch, config, accelerator):
    if accelerator.is_main_process:
        log_dir = config['scheme']['logdir']
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
        best_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_best.pth.tar')
        
        torch.save(state, latest_checkpoint)
        print(f"最新检查点已保存: {latest_checkpoint}")
        
        if is_best:
            shutil.copyfile(latest_checkpoint, best_checkpoint)
            print(f"新的最佳模型! 验证损失: {state['best_loss']:.6f}")
            print(f"最佳检查点已保存: {best_checkpoint}")

# --- 主函数 (多卡适配) ---
def main():
    args = parser.parse_args()
    
    # 初始化 Accelerator (使用简单配置避免之前的错误)
    mixed_precision = "no" if args.no_amp else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    
    if accelerator.is_main_process:
        print(f"=== 基于工作单卡版本的多卡训练 ===")
        print(f"可见GPU数量: {torch.cuda.device_count()}")
        print(f"进程数量: {accelerator.num_processes}")
        print(f"每个进程的设备: {accelerator.device}")
        print(f"混合精度类型: {accelerator.mixed_precision}")
        print("-" * 50)
    
    # 设置随机种子 (使用 Accelerate 的方式)
    #set_seed(args.seed)
    
    # 加载配置
    with open(args.config, 'r') as file:
        config = json.load(file)
    
    parameter = config['model']
    Data = config['data']
    scheme = config['scheme']
    log_dir = scheme['logdir']
    
    # 创建模型 (使用与单卡版本相同的参数)
    model = AutoencoderKL(
        in_channels=parameter['input_channels'],
        latent_channels=parameter['latent_channels']
    )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
    
    # 优化器 (与单卡版本相同)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=scheme['lr'], 
        weight_decay=scheme['weight_decay']
    )
    
    # 学习率调度器 (与单卡版本相同)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheme['gamma'])
    
    # 数据加载 (使用与单卡版本相同的数据集)
    train_dataset = MyVAEDataset(Data['data_train'], use_augmentation=False)
    val_dataset = MyVAEDatasetValidation(Data['data_val'], fixed_frame_policy='middle')
    
    if accelerator.is_main_process:
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
    
    # 批处理大小处理 (修复之前的除零错误)
    original_batch_size = scheme['batch_size']
    
    # 确保每个GPU至少分配到1个样本
    if original_batch_size < accelerator.num_processes:
        if accelerator.is_main_process:
            print(f"警告: 原始batch_size ({original_batch_size}) 小于GPU数量 ({accelerator.num_processes})")
            print(f"调整batch_size为 {accelerator.num_processes}")
        global_batch_size = accelerator.num_processes
        per_device_batch_size = 1
    else:
        per_device_batch_size = original_batch_size // accelerator.num_processes
        global_batch_size = per_device_batch_size * accelerator.num_processes
    
    if accelerator.is_main_process:
        print(f"全局batch size: {global_batch_size}")
        print(f"每个设备batch size: {per_device_batch_size}")
    
    # 数据加载器 (使用与单卡版本相同的配置)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        drop_last=True  # 确保所有进程批次大小一致
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False
    )
    
    # 使用 Accelerate 准备所有组件
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # 从检查点恢复 (适配多进程)
    start_epoch = scheme.get('start_epoch', 0)
    global best_loss_val
    
    if args.resume and os.path.isfile(args.resume):
        if accelerator.is_main_process:
            print(f"=> 正在加载检查点 '{args.resume}'")
        
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_epoch = checkpoint['epoch']
        
        # 获取unwrapped model来加载state_dict
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 处理 DDP 保存的 checkpoint 的 'module.' 前缀
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        unwrapped_model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        best_loss_val = checkpoint.get('best_loss', float("inf"))
        
        if accelerator.is_main_process:
            print(f"=> 已加载检查点 (epoch {checkpoint['epoch']})")
    
    # TensorBoard writer (只在主进程)
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard日志将保存到: {log_dir}")
    
    # 训练参数
    use_amp = not args.no_amp
    if accelerator.is_main_process:
        if use_amp:
            print("启用自动混合精度 (AMP)")
        else:
            print("禁用自动混合精度 (AMP)")
    
    # --- 训练循环 ---
    for epoch in range(start_epoch, scheme['epochs']):
        if accelerator.is_main_process:
            print(f"\n第 {epoch}/{scheme['epochs']-1} 轮")
            print("-" * 60)
        
        # 训练阶段 (使用与单卡版本相同的接口)
        train_loss, train_recon_loss, train_kl_loss = train_epoch(
            model, train_loader, optimizer, accelerator, scheme['beta'], 
            Data, args.print_freq, epoch, use_amp
        )
        
        # 验证阶段 (使用与单卡版本相同的接口) 
        val_loss, val_recon_loss, val_kl_loss = validate_epoch(
            model, val_loader, accelerator, scheme['beta'], Data, 
            args.print_freq, epoch
        )
        
        scheduler.step()
        
        # 记录和保存 (只在主进程)
        if accelerator.is_main_process and writer:
            writer.add_scalar("Training/Total_Loss", train_loss, epoch)
            writer.add_scalar("Validation/Total_Loss", val_loss, epoch)
            writer.add_scalar("Training/Recon_Loss", train_recon_loss, epoch) 
            writer.add_scalar("Validation/Recon_Loss", val_recon_loss, epoch)
            writer.add_scalar("Training/KL_Loss", train_kl_loss, epoch)
            writer.add_scalar("Validation/KL_Loss", val_kl_loss, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型 (只在主进程)
        if accelerator.is_main_process:
            is_best = val_loss < best_loss_val
            if is_best:
                best_loss_val = val_loss
            
            save_checkpoint({
                'epoch': epoch + 1,
                'config': config,
                'state_dict': accelerator.get_state_dict(model),
                'best_loss': best_loss_val,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, epoch, config, accelerator)
            
            print(f"训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 最佳验证损失: {best_loss_val:.6f}")
        
        # 确保所有进程同步
        accelerator.wait_for_everyone()
    
    if writer:
        writer.close()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("训练完成！")

# --- 训练函数 (与单卡版本接口完全一致) ---
# --- 训练函数 (修改版：增加重建和KL损失的实时打印) ---
def train_epoch(model, train_loader, optimizer, accelerator, beta, Data, print_freq, epoch, use_amp):
    model.train()
    
    # 定义用于跟踪指标的 AverageMeter
    batch_time = AverageMeter('时间', ':6.3f')
    total_losses = AverageMeter('总损失', ':.4e')
    recon_losses = AverageMeter('重建损失', ':.4e')
    kl_losses = AverageMeter('KL损失', ':.4e')

    # ### 关键修改: 将所有要打印的 meter 都加入 ProgressMeter ###
    if accelerator.is_main_process:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, total_losses, recon_losses, kl_losses], # <-- 把它们都放进来
            prefix=f"第 [{epoch}] 轮"
        )
    
    end = time.time()
    for step, images in enumerate(train_loader):
        # 数据处理
        image_norm = Normlize(images, Data["Total_std"], Data['Total_mean'])

        optimizer.zero_grad()
        
        # 前向传播
        with accelerator.autocast():
            recon_batch, posterior = model(sample=image_norm, sample_posterior=True, return_posterior=True)
            total_loss, recon_loss, kl_loss = loss_function(recon_batch, image_norm, posterior, kl_weight=beta)

        # 检查 NaN
        if torch.isnan(total_loss):
            if accelerator.is_main_process:
                print(f"!!! CRITICAL [Epoch {epoch}, Step {step}]: 损失值为 NaN！跳过此批次。")
            continue
        
        # 反向传播和优化
        accelerator.backward(total_loss)
        optimizer.step()
        
        # --- 同步所有损失值 ---
        # accelerator.gather 会收集所有进程的张量，然后我们计算均值
        total_loss_gathered = accelerator.gather(total_loss.detach()).mean()
        recon_loss_gathered = accelerator.gather(recon_loss.detach()).mean()
        kl_loss_gathered = accelerator.gather(kl_loss.detach()).mean()
        
        # --- 更新 AverageMeters ---
        # 使用全局 batch size 来更新，这样 .avg 才准确
        # accelerator.gather(images) 会收集所有进程的 images 张量
        global_batch_size = accelerator.gather(images).size(0)
        
        total_losses.update(total_loss_gathered.item(), global_batch_size)
        recon_losses.update(recon_loss_gathered.item(), global_batch_size)
        kl_losses.update(kl_loss_gathered.item(), global_batch_size)
        
        # 更新时间
        batch_time.update(time.time() - end)
        end = time.time()
        
        # ### 关键修改: progress.display() 会自动打印所有 meter ###
        if step % print_freq == 0 and accelerator.is_main_process:
            progress.display(step + 1)
            
    # 返回所有平均损失，以便 TensorBoard 记录
    return total_losses.avg, recon_losses.avg, kl_losses.avg

# --- 验证函数 (与单卡版本接口完全一致) ---
def validate_epoch(model, val_loader, accelerator, beta, Data, print_freq, epoch):
    model.eval()
    total_losses = AverageMeter('总损失', ':.5e')
    recon_losses = AverageMeter('重建损失', ':.5e')
    kl_losses = AverageMeter('KL损失', ':.5e')
    
    with torch.no_grad():
        for step, images in enumerate(val_loader):
            # 使用与单卡版本完全相同的数据处理
            image_norm = Normlize(images, Data["Total_std"], Data['Total_mean'])
            
            # 使用与单卡版本相同的验证时前向传播（注意这里不同）
            with accelerator.autocast():
                recon_batch, posterior = model(image_norm,sample_posterior=True, return_posterior=True)  # 验证时使用简化接口
                total_loss, recon_loss, kl_loss = loss_function(recon_batch, image_norm, posterior, kl_weight=beta)
            
            # 同步损失
            total_loss_gathered = accelerator.gather(total_loss.detach()).mean()
            recon_loss_gathered = accelerator.gather(recon_loss.detach()).mean()  
            kl_loss_gathered = accelerator.gather(kl_loss.detach()).mean()
            
            total_losses.update(total_loss_gathered.item(), images.size(0))
            recon_losses.update(recon_loss_gathered.item(), images.size(0))
            kl_losses.update(kl_loss_gathered.item(), images.size(0))
            
            # 绘图逻辑 (只在第一个step执行，只在主进程)
            if step == 10 :
                if accelerator.is_main_process:
                    # 使用与单卡版本相同的绘图逻辑
                    image_recon_cpu = Renormlize(
                        recon_batch[0].cpu(), 
                        dtstd=Data['Total_std'], 
                        dtmean=Data['Total_mean']
                    )
                    image_recon = torch.exp(image_recon_cpu).squeeze().detach().numpy()
                    image_truth = torch.exp(images[0].cpu().squeeze()).detach().numpy()
                    
                    Plot_radar(image_truth, image_recon, epoch, accelerator)
    
    if accelerator.is_main_process:
        print(f"  验证损失: {total_losses.avg:.6f}")
    
    return total_losses.avg, recon_losses.avg, kl_losses.avg

if __name__ == '__main__':
    main()