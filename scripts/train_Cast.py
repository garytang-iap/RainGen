#!/usr/bin/env python
# train_ddp.py - 基于train_accelerate.py改写的DDP版本

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import GradScaler, autocast
from collections import OrderedDict
import gc
import warnings
from pathlib import Path
from contextlib import nullcontext
from einops import rearrange
# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("TensorBoard not found. Install with 'pip install tensorboard'")
    TENSORBOARD_AVAILABLE = False

# 你的模型imports（保持不变）
from model.utils_diffusion import *
from model.autoencoder_kl import AutoencoderKL, LatteConditionalEncoder, LatteVAEDecoder
from model.diffusion_new import GaussianDiffusion
#from model.Latte_new import Latte
from model.ema import EMA
from model.radar_unet import SVDUNetBackbone # 【新增】SVD UNet 模型
# 警告过滤（保持不变）
warnings.filterwarnings(
    "ignore", 
    message="No device id is provided via `init_process_group` or `barrier `.", 
    category=UserWarning
)

# TF32设置（保持不变）
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch.set_float32_matmul_precision('high')
    print("Activated TF32 precision using torch.set_float32_matmul_precision('high').")
except AttributeError:
    print("Using legacy API to set TF32 precision.")
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
             torch.backends.cuda.matmul.fp32_precision = 'tf32'
        else:
             torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def setup_ddp():
    """初始化DDP"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()

def is_main_process():
    """是否主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    """获取进程rank"""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    """获取world size"""
    return dist.get_world_size() if dist.is_initialized() else 1

def parse_args():
    """解析命令行参数（保持不变）"""
    parser = argparse.ArgumentParser(description='DDP Latent Diffusion Model Training')
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--config', required=True, type=str, help='path to the configuration file')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # 新增：resume时的自定义学习率
    parser.add_argument('--resume-lr', default=None, type=float, 
                       help='Custom learning rate when resuming (overrides config and scheduler)')
    return parser.parse_args()

class AverageMeter:
    """计算和存储平均值和当前值（保持不变）"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """进度显示（保持不变）"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, eta=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if eta:
            entries.append(f"ETA: {eta}")
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def safe_load_model_state(model, checkpoint_state_dict, is_main):
    """安全加载模型状态（保持不变，修改is_main_process参数）"""
    try:
        model_state_dict = model.state_dict()
        
        # 处理DDP保存的'module.'前缀
        new_state_dict = OrderedDict()
        for k, v in checkpoint_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # 检查形状匹配
        shape_mismatches = []
        for key, param in model_state_dict.items():
            if key in new_state_dict:
                if param.shape != new_state_dict[key].shape:
                    shape_mismatches.append((key, param.shape, new_state_dict[key].shape))
        
        if shape_mismatches and is_main:
            print("⚠️ Shape mismatches found:")
            for key, model_shape, ckpt_shape in shape_mismatches[:3]:
                print(f"   {key}: model={model_shape}, checkpoint={ckpt_shape}")
        
        # 智能加载：只加载匹配的参数
        matched_state_dict = {k: v for k, v in new_state_dict.items() 
                             if k in model_state_dict and 
                             model_state_dict[k].shape == v.shape}
        
        missing_info = model.load_state_dict(matched_state_dict, strict=False)
        
        if is_main:
            loaded_count = len(matched_state_dict)
            total_count = len(model_state_dict)
            print(f"✅ Loaded {loaded_count}/{total_count} parameters successfully")
            
            if missing_info.missing_keys:
                print(f"ℹ️ {len(missing_info.missing_keys)} parameters initialized randomly")
                
        return True
    except Exception as e:
        if is_main:
            print(f"❌ Error loading model state: {e}")
        return False

def cleanup_memory():
    """强制清理Python和PyTorch的GPU显存（修改）"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Memory cleanup warning: {e}")

def reduce_tensor(tensor):
    """在所有进程间规约张量"""
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt

def validate_config(config):
    """验证配置文件的完整性（保持不变）"""
    required_sections = ['model_svd_unet', 'model_vae', 'data', 'diffusion', 'trainer']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # 设置默认值
    config['trainer'].setdefault('gradient_accumulation_steps', 1)
    config['trainer'].setdefault('max_grad_norm', 2.0)
    config['trainer'].setdefault('ema_decay', 0.999)
    config['trainer'].setdefault('warmup_epochs', 40)
    config['trainer'].setdefault('min_lr', 2e-6)
    config['diffusion'].setdefault('cond_scale', 1.0)
    config['diffusion'].setdefault('sampling_timesteps', 50)
    config['diffusion'].setdefault('ddim_sampling_eta', 0.0)
    
    return config
def decode_in_batches(vaedecoder, latent_5d, correction_factor, batch_size=5):
    """分批解码 - 保持5D格式"""
    B, T, C, H, W = latent_5d.shape
    decoded_frames = []
    
    for t_start in range(0, T, batch_size):
        t_end = min(t_start + batch_size, T)
        
        # 保持5D格式 [B, batch_size, C, H, W] 
        batch_5d = latent_5d[:, t_start:t_end]
        
        # 直接传5D给解码器
        decoded_batch = vaedecoder(batch_5d, correction_factor)
        decoded_frames.append(decoded_batch.cpu())
        
        del batch_5d, decoded_batch
        cleanup_memory()
    
    return torch.cat(decoded_frames, dim=1).to(latent_5d.device)
    
    return torch.cat(decoded_frames, dim=1).to(latent_5d.device)
def visualize_and_save_sample(epoch, diffusion_model_ema, vaedecoder, data_for_plot, config, device, use_amp=True):
    """
    可视化函数，现在接收 latent 作为输入。
    """
    if not is_main_process() or data_for_plot is None or vaedecoder is None:
        return
        
    cleanup_memory()
    
    if is_main_process():
        print("\n--- Generating visualization sample ---")
    
    try:
        scheme_config = config['diffusion']
        data_config = config['data']
        trainer_config = config['trainer']

        with torch.no_grad():
            condition_latent, target_latent_gt = data_for_plot
            condition_latent = condition_latent.to(device)
            target_latent_gt = target_latent_gt.to(device)
            
            # 1. Diffusion 采样，生成预测的 latent
            print(f"   - Sampling with condition shape: {condition_latent.shape}")
            with autocast(enabled=use_amp, device_type='cuda'):
                predicted_future_latent = diffusion_model_ema.sample(
                    condition=condition_latent,
                    frames=30,
                    cond_scale=scheme_config.get('cond_scale', 1.0),
                    sampling_timesteps=scheme_config.get('sampling_timesteps', 50),
                    ddim_sampling_eta=scheme_config.get('ddim_sampling_eta', 0.0)
                )
            
            # 2. 使用 VAE 解码器将 latents 转换回图像空间
            print("   - Decoding latents to image space...")
            correction_factor = 1.0/9.0
            # LatteVAEDecoder 应该能处理 (B, T, C, H, W) 的 5D 张量
            # 替换为：
            pred_img_decode = decode_in_batches(vaedecoder, predicted_future_latent, correction_factor)
            gt_img_decode = decode_in_batches(vaedecoder, target_latent_gt, correction_factor)
            input_img_decode = decode_in_batches(vaedecoder, condition_latent, correction_factor)
            
            # 3. 数据后处理，准备绘图
            print("   - Post-processing decoded images...")
            # 取批次中的第一个样本进行绘图
            pred_to_plot = torch.exp(
                Renormlize(pred_img_decode[0], data_config["Total_std"], data_config['Total_mean'])
            ).detach().cpu().numpy()
            
            gt_to_plot = torch.exp(
                Renormlize(gt_img_decode[0], data_config["Total_std"], data_config['Total_mean'])
            ).detach().cpu().numpy()
            
            input_to_plot = torch.exp(
                Renormlize(input_img_decode[0], data_config["Total_std"], data_config['Total_mean'])
            ).detach().cpu().numpy()

            # 4. 绘图与保存
            fig_dir = Path(trainer_config.get('train_fig', './outputs/figs/train'))
            fig_dir.mkdir(parents=True, exist_ok=True)
            save_path = fig_dir / f"epoch_{epoch+1}_sample.png"
            
            plot_long_forecast(
                GT_data=np.squeeze(gt_to_plot), # squeeze 掉通道维度
                DDPM_data=np.squeeze(pred_to_plot),
                input_data=np.squeeze(input_to_plot),
                save_dir=str(save_path)
            )
            print(f"   - Sample saved to: {save_path}")

    except Exception as e:
        print(f"❌ Visualization error: {e}")
        if args.debug:
            raise
def load_vae_decoder_for_visualization(config, device):
    if not is_main_process():
        return None

    parameter_vae = config['model_vae']
    print("Loading VAE weights to extract decoder for visualization...")
    
    # 完整实例化 VAE 以加载权重
    vae = AutoencoderKL(
        in_channels=parameter_vae['vae_input_channels'],
        out_channels=parameter_vae['vae_out_channels'],
        latent_channels=parameter_vae['latent_channels']
    )
    
    vae_checkpoint_path = parameter_vae['vae_checkpoint']
    if not os.path.exists(vae_checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint_path}")
        
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
    
    if 'state_dict' in vae_checkpoint: vae_state_dict = vae_checkpoint['state_dict']
    else: vae_state_dict = vae_checkpoint

    cleaned_state_dict = OrderedDict()
    for k, v in vae_state_dict.items():
        name = k.replace('_orig_mod.', '').replace('module.', '')
        cleaned_state_dict[name] = v
        
    vae.load_state_dict(cleaned_state_dict, strict=True)
    
    vae.eval().to(device)
    
    # 封装并返回 Decoder
    vaedecoder = LatteVAEDecoder(vae, scale=config['data'].get('scale_factor'))
    print("✅ VAE Decoder extracted successfully.")
    
    del vae
    cleanup_memory()
    return vaedecoder
    '''
def load_vae_model(config, device):
    """加载VAE模型（修改为DDP版本）"""
    parameter_vae = config['model_vae']
    
    vae = AutoencoderKL(
        in_channels=parameter_vae['vae_input_channels'], 
        out_channels=parameter_vae['vae_out_channels'],
        latent_channels=parameter_vae['latent_channels']
    )
    
    if not os.path.exists(parameter_vae['vae_checkpoint']):
        raise FileNotFoundError(f"VAE checkpoint not found: {parameter_vae['vae_checkpoint']}")
    
    vae_checkpoint = torch.load(parameter_vae['vae_checkpoint'], map_location='cpu')
    
    # 处理VAE checkpoint的参数名称问题
    vae_state_dict = vae_checkpoint['state_dict']
    cleaned_vae_state_dict = OrderedDict()
    
    for k, v in vae_state_dict.items():
        if k.startswith('_orig_mod.'):
            clean_key = k[10:]
        else:
            clean_key = k
        cleaned_vae_state_dict[clean_key] = v
    
    # 安全加载VAE
    try:
        vae.load_state_dict(cleaned_vae_state_dict, strict=True)
        if is_main_process():
            print("✅ VAE model loaded successfully")
    except Exception as e:
        if is_main_process():
            print(f"⚠️ VAE loading with strict=False: {e}")
        vae.load_state_dict(cleaned_vae_state_dict, strict=False)
    
    vae.eval()
    vae = vae.to(device)
    for p in vae.parameters(): 
        p.requires_grad = False
    
    del vae_checkpoint, cleaned_vae_state_dict
    return vae
'''
def main():
    args = parse_args()
    
    # DDP初始化
    local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed + local_rank)
        np.random.seed(args.seed + local_rank)
    
    # 验证配置文件
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, 'r') as file:
        config = json.load(file)
    
    config = validate_config(config)
    trainer_config = config['trainer']
    log_dir = trainer_config.get('log_dir', './outputs/logs')
    
    # 初始化TensorBoard（只在主进程）
    writer = None
    if is_main_process() and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard logs: {log_dir}")
    
    try:
        train_with_ddp(local_rank, world_size, device, config, args, writer)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.debug:
            raise
    finally:
        if writer:
            writer.close()
        cleanup_ddp()

def train_with_ddp(local_rank, world_size, device, config, args, writer):
    """DDP训练函数（修改为DDP版本）"""
    best_loss_val = float("inf")
    
    # ↓↓↓ 【修正】将这里的变量名修改正确 ↓↓↓
    svd_unet_config = config['model_svd_unet'] 
    data_config = config['data']
    scheme_config = config['diffusion']
    trainer_config = config['trainer']
    
    if is_main_process():
        print(f"=== SVD-UNet Radar Training (DDP) ===")
        print(f"Processes: {world_size}, Local rank: {local_rank}, Device: {device}")
        print(f"Mixed Precision: {'enabled' if not args.no_amp else 'disabled'}")
        print("-" * 50)
    '''
    # 模型初始化（保持不变）
    model = Latte(
        input_size=parameter_trans['input_size'],
        patch_size=parameter_trans['patch_size'],
        in_channels=parameter_trans['in_channels'],
        hidden_size=parameter_trans['hidden_size'],
        depth=parameter_trans['depth'],
        num_heads=parameter_trans['num_heads'],
        num_frames=parameter_trans['num_frames'],
        num_condition_frames=parameter_trans['num_condition_frames'],
        learn_sigma=parameter_trans.get('learn_sigma', True),
        enable_cfg=scheme_config.get('enable_cfg', True),
        cond_drop_prob=scheme_config.get('cond_drop_prob', 0.0)
    ).to(device)
    '''
    # 2. 【修改】实例化新的 SVDUNetBackbone 模型
    model = SVDUNetBackbone(
        input_size=svd_unet_config['input_size'],
        in_channels=svd_unet_config['in_channels'],
        model_channels=svd_unet_config['model_channels'],
        out_channels=svd_unet_config['out_channels'],
        attention_resolutions=svd_unet_config['attention_resolutions'],
        num_res_blocks=svd_unet_config['num_res_blocks'],
        channel_mult=svd_unet_config['channel_mult'],
        num_head_channels=svd_unet_config.get('num_head_channels', 64),
        #context_dim=svd_unet_config['context_dim'],
        transformer_depth=svd_unet_config.get('transformer_depth', 1),
        num_frames=data_config['output_frames'], 
    num_condition_frames=data_config['input_frames'],
        cond_drop_prob=scheme_config.get('cond_drop_prob', 0.1)
    ).to(device)
    # DDP包装
    model = DDP(model, device_ids=[local_rank],find_unused_parameters=False)
    
    # EMA初始化
    ema_decay = trainer_config.get('ema_decay', 0.999)
    ema_model = EMA(model.module, decay=ema_decay)
    
    # 优化器和调度器（保持不变）
    lr = trainer_config['lr']
    # 如果指定了resume学习率，覆盖配置
    # 如果指定了resume学习率，记录但先不修改
    resume_lr = args.resume_lr
    if resume_lr is not None and is_main_process():
        print(f"🔧 Will override peak LR: {lr:.2e} -> {resume_lr:.2e}")
    
    weight_decay = trainer_config.get('weight_decay', 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    warmup_epochs = trainer_config.get('warmup_epochs', 40)
    total_epochs = trainer_config['epochs']
    min_lr = trainer_config.get('min_lr', 2e-6)  # ← 移到这里
   # 创建调度器的函数
    # 创建调度器的函数
    def create_scheduler(peak_lr):
        """根据峰值学习率创建调度器"""
        # 重新设置optimizer的学习率为峰值学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = peak_lr
            
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
            return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        else:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

# 初始调度器
    scheduler = create_scheduler(lr)
    
    
    # 混合精度
    scaler = GradScaler(enabled=(not args.no_amp))
    use_amp = not args.no_amp
    
    if is_main_process():
        print(f"📚 LR: {lr:.2e}, Weight decay: {weight_decay}")
        print(f"📚 Warmup: {warmup_epochs} epochs, Total: {total_epochs} epochs")
    
    # 数据加载（修改为DDP版本）
    '''
    try:
        train_dataset = MyNewDataset(data_config['data_train'])
        val_dataset = MyNewDataset(data_config['data_val'])
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {e}")
    '''
     # 3. 【修改】使用新的数据集类加载预编码数据
    if is_main_process():
        print("--- Loading pre-encoded latent datasets ---")
    try:
        train_dataset = PreprocessedNpzDataset(data_config['preprocessed_data_train'])
        val_dataset = PreprocessedNpzDataset(data_config['preprocessed_data_val'])
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessed datasets: {e}")
    per_device_batch_size = trainer_config['batch_size']
    global_batch_size = per_device_batch_size * world_size
    
    if is_main_process():
        print(f"📊 Batch sizes: {per_device_batch_size} (per device), {global_batch_size} (global)")
    
    # 数据采样器和加载器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    num_workers = min(args.workers, os.cpu_count() // world_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=per_device_batch_size, 
        sampler=train_sampler,
        num_workers=10,  # 简化，避免多进程问题
        pin_memory=True,
        drop_last=True,
        
        prefetch_factor=3,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=per_device_batch_size, 
        sampler=val_sampler,
        num_workers=10,
        pin_memory=True,
        
        prefetch_factor=3,
    )
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Trainable parameters: {total_params:,}")
        effective_batch_size = global_batch_size * trainer_config.get('gradient_accumulation_steps', 1)
        print(f"📊 Effective batch size: {effective_batch_size}")
    
    # VAE加载
    #vae = load_vae_model(config, device)
    #conditional_encoder = LatteConditionalEncoder(vae, scale=data_config['scale_factor'])
    #vaedecoder = LatteVAEDecoder(vae, scale=data_config['scale_factor'])
    vaedecoder = load_vae_decoder_for_visualization(config, device)
    # Diffusion模型
    diffusion_model = GaussianDiffusion(
        model=model,
        timesteps=scheme_config['timesteps'],
        objective=scheme_config['objective'],
        beta_schedule=scheme_config.get('schedule', 'cosine'),
        offset_noise_strength=scheme_config.get('offset_noise_strength', 0.0),
    ).to(device)
    
    # EMA Diffusion模型
    diffusion_model_ema = GaussianDiffusion(
        model=ema_model.ema_model,
        timesteps=scheme_config['timesteps'],
        sampling_timesteps=scheme_config.get('sampling_timesteps', 50),
        objective=scheme_config['objective'],
        beta_schedule=scheme_config.get('schedule', 'cosine'),
        ddim_sampling_eta=scheme_config.get('ddim_sampling_eta', 0.0),
        offset_noise_strength=scheme_config.get('offset_noise_strength', 0.0),
    ).to(device)
    
    start_epoch = 0
    
    # 断点续训 - 修改这部分
    if args.resume and os.path.isfile(args.resume):
        if is_main_process():
            print(f"📂 Loading checkpoint: {args.resume}")
        
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(args.resume, map_location=map_location)
            start_epoch = checkpoint.get('epoch', 0)
            
            # 安全加载模型状态
            success = safe_load_model_state(model.module, checkpoint['state_dict'], is_main_process())
            if not success:
                raise RuntimeError("Failed to load model state")
            
            # EMA状态加载
            if 'ema_state_dict' in checkpoint:
                try:
                    ema_model.load_state_dict(checkpoint['ema_state_dict'])
                    if is_main_process():
                        print("✅ EMA state loaded")
                except Exception as e:
                    if is_main_process():
                        print(f"⚠️ EMA sync from main model: {e}")
                    ema_model.ema_model.load_state_dict(model.module.state_dict())
            
            # 如果指定了新的峰值学习率，重建scheduler和optimizer
            if resume_lr is not None:
                if is_main_process():
                    print(f"🔧 Rebuilding scheduler with new peak LR: {resume_lr:.2e}")
                
                # 重新设置optimizer的学习率为新的峰值
                for param_group in optimizer.param_groups:
                    param_group['lr'] = resume_lr
                
                # 用新的峰值学习率重建调度器
                scheduler = create_scheduler(resume_lr)
                
                # 将调度器快进到当前epoch的状态
                for _ in range(start_epoch):
                    scheduler.step()
                
                # 计算当前应该的学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                if is_main_process():
                    print(f"📊 Scheduler fast-forwarded to epoch {start_epoch}")
                    print(f"📊 Current LR after fast-forward: {current_lr:.2e}")
                    
            else:
                # 正常加载optimizer和scheduler状态
                if 'optimizer' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        if is_main_process():
                            print("✅ Optimizer state loaded")
                    except Exception as e:
                        if is_main_process():
                            print(f"⚠️ Optimizer state incompatible: {e}")
                
                if 'scheduler' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                        if is_main_process():
                            print("✅ Scheduler state loaded")
                    except Exception as e:
                        if is_main_process():
                            print(f"⚠️ Scheduler state incompatible: {e}")
            
            if is_main_process():
                best_loss_val = checkpoint.get('best_val_loss', float('inf'))
                current_lr = optimizer.param_groups[0]['lr']
                print(f"🔄 Resume from epoch {start_epoch}")
                print(f"📊 Current LR: {current_lr:.2e}")
                print(f"🏆 Best loss: {best_loss_val:.4e}")
            
            del checkpoint
        except Exception as e:
            if is_main_process():
                print(f"❌ Resume failed: {e}")
            raise
    # 训练循环（主要修改）
    global_step = start_epoch * len(train_loader)
    gradient_accumulation_steps = trainer_config.get('gradient_accumulation_steps', 1)
    
    for epoch in range(start_epoch, trainer_config['epochs']):
        # 设置epoch用于DistributedSampler
        train_sampler.set_epoch(epoch)
        
        if is_main_process():
            print(f"\n--- Epoch {epoch+1}/{trainer_config['epochs']} ---")
        
        model.train()
        train_meters = {
            "loss": AverageMeter('Loss', ':.4e'),
            "model_std": AverageMeter('ModelStd', ':.4f'),
            "target_std": AverageMeter('TargetStd', ':.4f'),
            "time": AverageMeter('Time', ':6.3f'),
            "data_time": AverageMeter('DataTime', ':6.3f')
        }
        
        if is_main_process():
            progress = ProgressMeter(len(train_loader), train_meters.values(), 
                                   prefix=f"Epoch: [{epoch+1}]")
        
        end = time.time()
        
        #for i, (images, label) in enumerate(train_loader):
        for i, (condition_latent, target_latent) in enumerate(train_loader): # <-- **变量名改变**
            #if i>5:continue
            data_time = time.time() - end
            train_meters["data_time"].update(data_time)
            
            condition_latent = condition_latent.to(device, non_blocking=True)
            target_latent = target_latent.to(device, non_blocking=True)
            '''
            # VAE编码
            with torch.no_grad():
                try:
                    #image_norm = Normlize(images, data_config["Total_std"], data_config['Total_mean'])
                    label_norm = Normlize(label, data_config["Total_std"], data_config['Total_mean'])
                    #encoded_condition = conditional_encoder(image_norm)
                    target_latent = conditional_encoder(label_norm)
                except Exception as e:
                    print(f"VAE encoding error: {e}")
                    continue'''
           
            
            # 前向传播
            with autocast(enabled=use_amp,device_type='cuda'):
                try:
                    loss, model_std, target_std = diffusion_model(img=target_latent, condition=condition_latent)
                
                    # 检查loss有效性
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ Invalid loss detected: {loss.item()}, skipping")
                        continue
                    
                    # 缩放loss用于梯度累积
                    loss = loss / gradient_accumulation_steps
                        
                except Exception as e:
                    print(f"Forward pass error: {e}")
                    continue

            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (i + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                        trainer_config.get('max_grad_norm', 2.0))
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 更新EMA
                ema_model.update()
                global_step += 1

            # 统计与日志（修改为DDP版本）
            if (i + 1) % gradient_accumulation_steps == 0:
                try:
                    # 规约loss到所有进程
                    loss_for_log = loss * gradient_accumulation_steps  # 恢复原始loss
                    loss_gathered = reduce_tensor(loss_for_log.detach())
                    model_std_gathered = reduce_tensor(model_std.detach())
                    target_std_gathered = reduce_tensor(target_std.detach())
                    
                    train_meters["loss"].update(loss_gathered.item())
                    train_meters["model_std"].update(model_std_gathered.item())
                    train_meters["target_std"].update(target_std_gathered.item())
                    
                    # TensorBoard记录（只在主进程）
                    if is_main_process() and writer:
                        log_data = {
                            "train/loss_step": loss_gathered.item(),
                            "train/model_std_step": model_std_gathered.item(),
                            "train/target_std_step": target_std_gathered.item(),
                            "train/learning_rate": optimizer.param_groups[0]['lr']
                        }
                        for key, value in log_data.items():
                            writer.add_scalar(key, value, global_step)
                except Exception as e:
                    print(f"Logging error: {e}")
            
            batch_time = time.time() - end
            train_meters["time"].update(batch_time)
            end = time.time()
            
            if i % args.print_freq == 0 and is_main_process():
                progress.display(i + 1)
        
        # Epoch结束
        if is_main_process():
            print(f"✅ Epoch {epoch+1} completed: Loss {train_meters['loss'].avg:.4e}")
            
            if writer:
                log_data_epoch = {
                    "train/loss_epoch": train_meters['loss'].avg,
                    "train/model_std_epoch": train_meters['model_std'].avg,
                    "train/target_std_epoch": train_meters['target_std'].avg,
                    "epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                for key, value in log_data_epoch.items():
                    writer.add_scalar(key, value, global_step)

        scheduler.step()
        cleanup_memory()
        
        # 验证阶段（修改为DDP版本）
        model.eval()
        val_losses_ema = AverageMeter('ValLossEMA', ':.4e')
        data_for_plot = None
        
        with torch.no_grad():
            #for i, (images_val, label_val) in enumerate(val_loader): 
            for i, (condition_latent_val, target_latent_val) in enumerate(val_loader):
                #if i>5:continue
                condition_latent_val = condition_latent_val.to(device, non_blocking=True)
                target_latent_val = target_latent_val.to(device, non_blocking=True)
                
                
                        
                with autocast(enabled=use_amp, device_type='cuda'):
                    loss_v_ema, _, _ = diffusion_model_ema(img=target_latent_val, condition=condition_latent_val)
                if not (torch.isnan(loss_v_ema) or torch.isinf(loss_v_ema)):
                    loss_ema_gathered = reduce_tensor(loss_v_ema.detach())
                    val_losses_ema.update(loss_ema_gathered.item(), condition_latent_val.size(0))
                    
                    
        
        # 验证结束
        if is_main_process():
            print(f"🔍 Epoch {epoch+1} Validation: {val_losses_ema.avg:.4e}")
            
            if writer:
                writer.add_scalar("validation/loss_ema", val_losses_ema.avg, global_step)
        
        # --- 修正后的可视化采样部分 ---
        if is_main_process() and ((epoch + 1) % 5 == 0 or epoch == 0): # 只在每10个 epoch 结束时可视化
            if vaedecoder is None:
                vaedecoder = load_vae_decoder_for_visualization(config, device)
            # 从验证集中获取一个批次用于可视化
            # 随机选择验证集中的一个批次进行可视化
            random_idx = random.randint(0, len(val_loader) - 1)
            
            # 获取随机批次
            for i, (vis_cond_latent, vis_target_latent) in enumerate(val_loader):
                if i == random_idx:
                    break
            
            visualize_and_save_sample(
                epoch=epoch, 
                diffusion_model_ema=diffusion_model_ema,
                vaedecoder=vaedecoder, 
                data_for_plot=(vis_cond_latent[:1], vis_target_latent[:1]), # 取第一个样本
                config=config, 
                device=device,
                use_amp=use_amp
            )

            
                
           
                
        # 保存检查点（只在主进程）
        if is_main_process():
            is_best = val_losses_ema.avg < best_loss_val
            if is_best:
                best_loss_val = val_losses_ema.avg
            
            try:
                save_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),  # 注意DDP的module属性
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_loss_val,
                    'config': config,
                    'global_step': global_step
                }
                
                checkpoint_dir = Path(trainer_config.get('checkpoint_dir', './outputs/checkpoints'))
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                latest_path = checkpoint_dir / 'checkpoint_latest.pth.tar'
                torch.save(save_state, latest_path)
                
                if is_best:
                    best_path = checkpoint_dir / "checkpoint_best.pth.tar"
                    torch.save(save_state, best_path)
                    print(f"🏆 New best model: {best_loss_val:.4e}")
                    
            except Exception as e:
                print(f"❌ Checkpoint save failed: {e}")

    if is_main_process():
        print("🎉 Training completed successfully!")

if __name__ == '__main__':
    main()