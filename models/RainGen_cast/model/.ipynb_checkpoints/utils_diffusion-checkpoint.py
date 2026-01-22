import math
import os
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
import random
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler

import numpy as np
from torchvision import datasets, transforms
import torch.distributed as dist
import shutil
import matplotlib.pyplot as plt
from enum import Enum
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 添加用于绘制概率集合结果的函数
def plot_prob_ensemble(GT_data, ensemble_data, spread_data, input_data, file_name, save_dir, time_indices=[9]):
    """
    Plot probabilistic ensemble forecast results and spread analysis
    
    :param GT_data: Ground truth data, shape [T, H, W]
    :param ensemble_data: Ensemble mean prediction, shape [T, H, W]
    :param spread_data: Ensemble standard deviation, shape [T, H, W]
    :param input_data: Input data sequence, shape [T, H, W]
    :param file_name: Source file name (for title)
    :param save_dir: Output directory path
    :param time_indices: List of time steps to visualize
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    # Radar reflectivity colormap
    cmap = ListedColormap([
        '#FFFFFF', '#DDF0FF', '#CCE0FF', '#AACEEB', '#99BBDA',
        '#7799CC', '#5588BB', '#4466AA', '#224499', '#002277',
        '#000066', '#99FF99', '#88EE77', '#77DD55', '#66CC44',
        '#55BB22', '#44AA11', '#227700', '#DDDD00', '#FFCC00',
        '#FFBB00', '#FFAA00', '#FF8800', '#FF6600', '#FF4400',
        '#FF2200', '#FF0000', '#EE0000', '#DD0000', '#CC0000',
        '#BB0000', '#AA0000', '#990000', '#770000', '#550000'
    ])

    for time_idx in time_indices:
        if time_idx >= len(GT_data):
            continue

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f"Probabilistic Ensemble Forecast - {file_name} - Time Step {time_idx}", fontsize=16)

        # 1. Input Data (last frame)
        im1 = axes[0, 0].imshow(input_data[-1], cmap=cmap, vmin=0, vmax=35)
        axes[0, 0].set_title(f'Input Data (t={len(input_data)-1})')
        fig.colorbar(im1, ax=axes[0, 0])

        # 2. Ground Truth
        im2 = axes[0, 1].imshow(GT_data[time_idx], cmap=cmap, vmin=0, vmax=35)
        axes[0, 1].set_title(f'Ground Truth (t={time_idx})')
        fig.colorbar(im2, ax=axes[0, 1])

        # 3. Ensemble Forecast
        im3 = axes[1, 0].imshow(ensemble_data[time_idx], cmap=cmap, vmin=0, vmax=35)
        axes[1, 0].set_title(f'Ensemble Forecast (t={time_idx})')
        fig.colorbar(im3, ax=axes[1, 0])

        # 4. Spread Analysis
        spread_max = np.max(spread_data[time_idx]) * 1.2
        im4 = axes[1, 1].imshow(spread_data[time_idx], cmap='viridis', vmin=0, vmax=spread_max)
        axes[1, 1].set_title(f'Ensemble Spread (Std Dev, t={time_idx})')
        fig.colorbar(im4, ax=axes[1, 1])

        # Calculate metrics
        mae = np.mean(np.abs(ensemble_data[time_idx] - GT_data[time_idx]))
        max_diff = np.max(np.abs(ensemble_data[time_idx] - GT_data[time_idx]))
        csi2 = CSI(GT_data[time_idx], ensemble_data[time_idx], 2.0)
        csi8 = CSI(GT_data[time_idx], ensemble_data[time_idx], 8.0)

        # Add metrics footer
        plt.figtext(0.5, 0.01,
                   f'Metrics: MAE={mae:.4f}, Max Error={max_diff:.4f}, CSI2={csi2:.4f}, CSI8={csi8:.4f}\n'
                   f'Spread Stats: Mean={np.mean(spread_data[time_idx]):.4f}, Max={np.max(spread_data[time_idx]):.4f}',
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        # Save visualization
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(save_dir, dpi=200, bbox_inches='tight')
        plt.close(fig)
def identity(x):
    return x

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    """函数的作用是检查一个数字 num 是否是一个完全平方数（即它是否有整数平方根）。"""
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    """函数的作用是将一个数字 num 按照指定的除数 divisor 分成多个组，每组的大小为 divisor，如果有剩余，则将剩余的部分作为一个额外的组返回"""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    """将图像 image 转换为指定模式 img_type，如果图像的当前模式与目标模式不匹配的话。如果图像已经是目标模式，则直接返回原图像"""
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def pack_one_with_inverse(x, pattern):
    """将一个张量 x 根据 pattern 进行打包（通过 pack 函数），并返回一个打包后的张量 packed 和一个“逆”函数 inverse。该 inverse 函数能够解包张量并还原到原始结构"""
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

# normalization functions

def normalize_to_neg_one_to_one(img):
    """将图像 img 的像素值归一化到 [-1, 1] 的范围，通常用于神经网络训练中的图像预处理。"""
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """将图像张量 t 从 [-1, 1] 的范围反归一化到 [0, 1] 的范围。"""
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    """生成一个指定形状 shape 的张量，元素值是从均匀分布中随机采样的（范围为 [0, 1]）。生成的张量将在指定的 device（如 CPU 或 GPU）上"""
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    """生成一个与给定形状 shape 相同的布尔型张量（或概率张量），值为 1 的概率为 prob，其他部分为 0。即生成一个掩码张量，控制在某些位置保留原数据。"""
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    

def project(x, y):
    """将两个向量 x 和 y 进行投影。首先将 y 归一化，然后将 x 在 y 的方向上投影，最后计算 x 与 y 的正交部分。"""
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

def condition_random_drop(total_count, ratio):
    # 根据比例计算需要选择的整数个数
    n=total_count
    p=ratio
    mask= torch.bernoulli(torch.full((n,), p))

    return mask


def loss_function(recon_x, x, posterior, kl_weight=1.0):
    # 重构损失：使用L1损失
    recon_loss = torch.nn.functional.l1_loss(recon_x, x, reduction='sum') / recon_x.size(0)
    
    # KL散度
    kl_loss = posterior.kl()
    kl_loss = kl_loss.mean()
    
    # 总损失
    loss = recon_loss + kl_weight * kl_loss
    return loss, recon_loss, kl_loss




class Latent_Dataset(Dataset):
    def __init__(self, data_dir, is_crop=True,size=160,input_length=5, label_length=10):
        """
        Args:
            data_dir (str): 数据目录路径
            input_length (int): 输入序列的长度，默认为5
            label_length (int): 标签序列的长度，默认为10
        """
        self.data_dir = data_dir
        # 按时间顺序排列文件
        self.data = np.sort(os.listdir(data_dir))
        
        self.input_length = input_length    # 输入帧数
        self.label_length = label_length    # 标签帧数
        self.sequence_length = input_length + label_length  # 总帧数
        self.is_crop=is_crop
        # 确保有足够的连续帧
        self.size = len(self.data) - (self.sequence_length - 1)
        self.crop_size=size
        
        if self.size <= 0:
            raise ValueError(
                f"数据量不足！需要至少 {self.sequence_length} 帧连续数据，"
                f"但目录中只有 {len(self.data)} 个文件"
            )
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        seed = torch.random.seed()
        crop=transforms.RandomCrop(self.crop_size)
        try:
            # 读取连续帧数据
            frames = []
            for i in range(self.sequence_length):
                data_path = self.data[index + i]
                frame = np.load(os.path.join(self.data_dir, data_path))['data']
                frames.append(frame)
            
            # 将数据堆叠并转换为tensor
            frames = np.stack(frames)
            frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1, H, W]
            if self.is_crop:
                torch.random.manual_seed(seed)
                frames=crop(frames)
            
            # 分离输入和标签
            input_data = frames[:self.input_length]     # [input_length, 1, H, W]
            label_data = frames[self.input_length:]     # [label_length, 1, H, W]
            
            return input_data, label_data
            
        except Exception as e:
            print(f"Error loading sequence starting from index {index}")
            print(f"First file in sequence: {os.path.join(self.data_dir, self.data[index])}")
            print(e)
            raise e


class MyNewDataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir=data_dir
        self.data=os.listdir(data_dir)
        self.size=len(self.data)
        
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        seed = torch.random.seed()
        crop=transforms.RandomCrop(160)
        
        data_path=self.data[index]
        try:
            imgs=torch.tensor(np.load(os.path.join(self.data_dir,data_path))['image'],dtype=torch.float32)
            torch.random.manual_seed(seed)
            imgs_crop=crop(imgs).unsqueeze(1)

            lbs=torch.tensor(np.load(os.path.join(self.data_dir,data_path))['label'],dtype=torch.float32)
            torch.random.manual_seed(seed)
            lbs_crop=crop(lbs).unsqueeze(1)
        except Exception as e:
            # 如果捕获到异常，打印出错误信息以及问题文件的路径
            print(f"Error loading file: {os.path.join(self.data_dir, data_path)}")
            print(e)
        
        
        
        return imgs_crop,lbs_crop
    
def Normlize(data,dtstd,dtmean):
    data_norm=(data-dtmean)/dtstd
    return data_norm

def Renormlize(data_norm,dtstd,dtmean):
    data=data_norm*(dtstd)+dtmean
    return data


def save_checkpoint(state, is_best_val, is_best_train, filename='checkpoint.pth.tar', best_model_path='model_best.pth.tar', train_best_path='train_best.pth.tar'):
    """
    Saves model checkpoint.

    Args:
        state (dict): Contains model's state_dict, optimizer_state_dict, epoch, etc.
        is_best_val (bool): True if this is the best model based on validation loss.
        is_best_train (bool): True if this is the best model based on training loss.
        filename (str): Filename for the current checkpoint.
        best_model_path (str): Path to save the best validation model.
        train_best_path (str): Path to save the best training model.
    """
    # 确保目录存在
    checkpoint_dir = os.path.dirname(filename)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_dir = os.path.dirname(best_model_path)
    if best_model_dir and not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir, exist_ok=True)

    train_best_dir = os.path.dirname(train_best_path)
    if train_best_dir and not os.path.exists(train_best_dir):
        os.makedirs(train_best_dir, exist_ok=True)

    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")
    if is_best_val:
        shutil.copyfile(filename, best_model_path)
        print(f"Best validation model saved to {best_model_path}")
    if is_best_train: # 你原来的参数名是 loss_best，这里改为 is_best_train 更清晰
        shutil.copyfile(filename, train_best_path)
        print(f"Best training model saved to {train_best_path}")

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def MAE_loss(Predict,GT):
    '''预测值和真实值算MAE'''
    error_abs=np.abs(Predict-GT)
    MAE=error_abs.mean(0).mean(0)
    return MAE

def Plot_radar(image,image_recon,epoch):

    levs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    cols = ["#D3D3D3","#01a0f6", "#00ecec", "#6dfa3d", "#00D806", "#019000", "#FFFF00", "#e7c000", "#FF9000", "#FF0000", "#d60000", "#C00000", "#e4007e", "#9600b4", "#AD90F0"]
    cmap = ListedColormap(cols,N=15)
    MAE_recon=MAE_loss(image,image_recon)
    fig,ax=plt.subplots(1,2,figsize=(5,10))
    im=ax[0].imshow(image,cmap=cmap,vmin=0.0,vmax=70.0)
    ax[0].set_title("Truth",fontsize=19)
    ax[1].imshow(image_recon,cmap=cmap,vmin=0.0,vmax=70.0)
    ax[1].set_title("Recon",fontsize=19)
    ax[1].text(0.1,60,"MAE={:.2f}".format(MAE_recon))
    for axes in ax.flat:
        # 隐藏轴线
        for spine in axes.spines.values():
            spine.set_visible(False)
        # 隐藏刻度标签和刻度线
        axes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    for axes in ax.flat:
        axes.grid(linestyle='-.')
    cbar = fig.colorbar(im, ax=ax,orientation='horizontal', shrink=0.5,pad=0.05,extend='max')
    plt.savefig("/data1/share/tangxiao1/RadarModel/code/VAE/fig/evaluation_{}.png".format(epoch))
    plt.close()


def prep_clf(obs,pre, threshold=0.1):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN 
    '''
    #根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives


def precision(obs, pre, threshold=0.1):
    '''
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, pre, threshold=0.1):
    '''
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FN)


def ACC(obs, pre, threshold=0.1):
    '''
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)

def FSC(obs, pre, threshold=0.1):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))

def MAE_loss(Predict,GT):
    '''预测值和真实值算MAE'''
    error_abs=np.abs(Predict-GT)
    MAE=error_abs.mean(1).mean(1)
    return MAE

def CSI(obs, pre, threshold=0.1):
    
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return hits/(hits + falsealarms + misses)

def plot_Rainfall(GT_data,DDPM_data,input_data,save_dir=None):
    levs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    cols = ["#D3D3D3","#01a0f6", "#00ecec", "#6dfa3d", "#00D806", "#019000", "#FFFF00", "#e7c000", "#FF9000", "#FF0000", "#d60000", "#C00000", "#e4007e", "#9600b4", "#AD90F0"]
    cmap = ListedColormap(cols,N=15)
    #norm = BoundaryNorm(levs,cmap.N, clip=False)
    #CRPS_DDPM=cal_crps_max(DDPM_data,GT_data)
    MAE_DDPM=MAE_loss(DDPM_data,GT_data)
    fig,ax=plt.subplots(3,5,figsize=(25,15))
    im=ax[0,0].imshow(GT_data[1],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[0,0].set_ylabel("TARGET",fontsize=19)
    ax[0,0].set_title("T+6min",fontsize=19)
    ax[0,1].imshow(GT_data[3],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[0,1].set_title("T+24min",fontsize=19)
    ax[0,2].imshow(GT_data[5],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[0,2].set_title("T+36min",fontsize=19)
    ax[0,3].imshow(GT_data[7],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[0,3].set_title("T+48min",fontsize=19)
    ax[0,4].imshow(GT_data[9],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[0,4].set_title("T+60min",fontsize=19)
    
    ax[1,0].imshow(DDPM_data[1],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[1,0].text(0.1,60,"MAE={:.2f}".format(MAE_DDPM[1]))
    ax[1,0].text(0.1,80,"CSI2={:.2f}".format(CSI(GT_data[1],DDPM_data[1],2.0)))
    ax[1,0].text(0.1,100,"CSI8={:.2f}".format(CSI(GT_data[1],DDPM_data[1],8.0)))
    ax[1,0].set_ylabel("Classifer-free guidance",fontsize=19)
    ax[1,1].imshow(DDPM_data[3],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[1,1].text(0.1,60,"MAE={:.2f}".format(MAE_DDPM[3]))
    ax[1,1].text(0.1,80,"CSI2={:.2f}".format(CSI(GT_data[3],DDPM_data[3],2.0)))
    ax[1,1].text(0.1,100,"CSI8={:.2f}".format(CSI(GT_data[3],DDPM_data[3],8.0)))
    ax[1,2].imshow(DDPM_data[5],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[1,2].text(0.1,60,"MAE={:.2f}".format(MAE_DDPM[5]))
    ax[1,2].text(0.1,80,"CSI2={:.2f}".format(CSI(GT_data[5],DDPM_data[5],2.0)))
    ax[1,2].text(0.1,100,"CSI8={:.2f}".format(CSI(GT_data[5],DDPM_data[5],8.0)))
    ax[1,3].imshow(DDPM_data[7],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[1,3].text(0.1,60,"MAE={:.2f}".format(MAE_DDPM[7]))
    ax[1,3].text(0.1,80,"CSI2={:.2f}".format(CSI(GT_data[7],DDPM_data[7],2.0)))
    ax[1,3].text(0.1,100,"CSI8={:.2f}".format(CSI(GT_data[7],DDPM_data[7],8.0)))
    ax[1,4].imshow(DDPM_data[9],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[1,4].text(0.1,60,"MAE={:.2f}".format(MAE_DDPM[9]))
    ax[1,4].text(0.1,80,"CSI2={:.2f}".format(CSI(GT_data[9],DDPM_data[9],2.0)))
    ax[1,4].text(0.1,100,"CSI8={:.2f}".format(CSI(GT_data[9],DDPM_data[9],8.0)))
    ax[2,0].imshow(input_data[0],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[2,0].set_ylabel("Input",fontsize=19)
    ax[2,1].imshow(input_data[1],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[2,2].imshow(input_data[2],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[2,3].imshow(input_data[3],cmap=cmap,vmin=0.0,vmax=70.0)
    ax[2,4].imshow(input_data[4],cmap=cmap,vmin=0.0,vmax=70.0)
    

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for axes in ax.flat:
        # 隐藏轴线
        for spine in axes.spines.values():
            spine.set_visible(False)
        # 隐藏刻度标签和刻度线
        axes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    for axes in ax.flat:
        axes.grid(linestyle='-.')
    cbar = fig.colorbar(im, ax=ax,orientation='horizontal', shrink=0.5,pad=0.05,extend='max')
    plt.savefig(save_dir)
    plt.close()

def plot_members_comparison(GT_data, all_members_data, ensemble_data1, ensemble_data2, input_data, file_name, save_dir, time_idx=-1):
    """
    绘制所有成员和集合预报在特定时刻的比较图
    
    参数:
    GT_data: 真实数据，numpy数组
    all_members_data: 所有成员的预报结果列表，每个元素是一个numpy数组
    ensemble_data1: 方法1的集合预报结果 (先平均后指数)
    ensemble_data2: 方法2的集合预报结果 (先指数后平均)
    input_data: 输入数据
    file_name: 输出文件名前缀
    save_dir: 保存目录
    time_idx: 要绘制的时间步索引，默认为-1表示最后一个时间步
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    import os
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # 色标设置
    levs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    cols = ["#D3D3D3","#01a0f6", "#00ecec", "#6dfa3d", "#00D806", "#019000", "#FFFF00", "#e7c000", "#FF9000", "#FF0000", "#d60000", "#C00000", "#e4007e", "#9600b4", "#AD90F0"]
    cmap = ListedColormap(cols, N=15)
    
    # 获取成员数量
    n_members = len(all_members_data)
    
    # 计算行数和列数 (每行最多4个面板)
    n_cols = min(4, n_members + 3)  # +3 是为了真实值、两种集合方法和输入
    n_rows = (n_members + 3 + n_cols - 1) // n_cols
    
    # 创建足够大的图表
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 绘制真实值
    im = axes[0].imshow(GT_data[time_idx], cmap=cmap, vmin=0.0, vmax=70.0)
    axes[0].set_title("Ground Truth", fontsize=12)
    
    # 绘制两种集合方法结果
    axes[1].imshow(ensemble_data1[time_idx], cmap=cmap, vmin=0.0, vmax=70.0)
    mae1 = np.abs(ensemble_data1[time_idx] - GT_data[time_idx]).mean()
    csi2_1 = CSI(GT_data[time_idx], ensemble_data1[time_idx], 2.0)
    csi8_1 = CSI(GT_data[time_idx], ensemble_data1[time_idx], 8.0)
    axes[1].set_title(f"Ensemble Method 1\nMAE={mae1:.2f}, CSI2={csi2_1:.2f}, CSI8={csi8_1:.2f}", fontsize=10)
    
    axes[2].imshow(ensemble_data2[time_idx], cmap=cmap, vmin=0.0, vmax=70.0)
    mae2 = np.abs(ensemble_data2[time_idx] - GT_data[time_idx]).mean()
    csi2_2 = CSI(GT_data[time_idx], ensemble_data2[time_idx], 2.0)
    csi8_2 = CSI(GT_data[time_idx], ensemble_data2[time_idx], 8.0)
    axes[2].set_title(f"Ensemble Method 2\nMAE={mae2:.2f}, CSI2={csi2_2:.2f}, CSI8={csi8_2:.2f}", fontsize=10)
    
    # 绘制最后一帧输入
    axes[3].imshow(input_data[-1], cmap=cmap, vmin=0.0, vmax=70.0)
    axes[3].set_title("Last Input Frame", fontsize=12)
    
    # 绘制每个成员的预报
    for i, member_data in enumerate(all_members_data):
        idx = i + 4  # 前4个位置已被使用
        if idx < len(axes):
            axes[idx].imshow(member_data[time_idx], cmap=cmap, vmin=0.0, vmax=70.0)
            # 计算评估指标
            mae = np.abs(member_data[time_idx] - GT_data[time_idx]).mean()
            csi2 = CSI(GT_data[time_idx], member_data[time_idx], 2.0)
            csi8 = CSI(GT_data[time_idx], member_data[time_idx], 8.0)
            axes[idx].set_title(f"Member {i}\nMAE={mae:.2f}, CSI2={csi2:.2f}, CSI8={csi8:.2f}", fontsize=10)
    
    # 隐藏多余的子图
    for i in range(n_members + 4, len(axes)):
        axes[i].axis('off')
    
    # 格式化所有子图
    for ax in axes:
        if not ax.get_axes_locator():
            # 隐藏轴线
            for spine in ax.spines.values():
                spine.set_visible(False)
            # 隐藏刻度标签和刻度线
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                           labelbottom=False, labelleft=False)
            # 添加网格线
            ax.grid(linestyle='-.')
    
    # 添加颜色条
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # 设置整体标题
    min_time = time_idx * 6
    plt.suptitle(f"Members Comparison at T+{min_time}min - {file_name}", fontsize=16)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    comparison_dir = os.path.join(os.path.dirname(save_dir), "members_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(f"{comparison_dir}/{file_name}_T{min_time}min_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_ensemble_spread(GT_data, all_members_data, ensemble_data, input_data, file_name, save_dir, time_idx=-1):
    """
    绘制集合预报的离散度和成员分布
    
    参数:
    GT_data: 真实数据，numpy数组
    all_members_data: 所有成员的预报结果列表，每个元素是一个numpy数组
    ensemble_data: 集合预报结果
    input_data: 输入数据
    file_name: 输出文件名前缀
    save_dir: 保存目录
    time_idx: 要绘制的时间步索引，默认为-1表示最后一个时间步
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    import os
    
    # 获取成员数量
    n_members = len(all_members_data)
    
    # 计算集合离散度 (标准差)
    all_members_stack = np.stack(all_members_data)
    ensemble_std = np.std(all_members_stack, axis=0)[time_idx]
    
    # 计算误差范围 (最大值和最小值之间的差异)
    ensemble_max = np.max(all_members_stack, axis=0)[time_idx]
    ensemble_min = np.min(all_members_stack, axis=0)[time_idx]
    ensemble_range = ensemble_max - ensemble_min
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 色标设置
    levs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    cols = ["#D3D3D3","#01a0f6", "#00ecec", "#6dfa3d", "#00D806", "#019000", "#FFFF00", "#e7c000", "#FF9000", "#FF0000", "#d60000", "#C00000", "#e4007e", "#9600b4", "#AD90F0"]
    cmap = ListedColormap(cols, N=15)
    
    # 第一行: 真实值、集合平均和最后一帧输入
    # 绘制真实值
    axes[0, 0].imshow(GT_data[time_idx], cmap=cmap, vmin=0.0, vmax=70.0)
    axes[0, 0].set_title("Ground Truth", fontsize=12)
    
    # 绘制集合平均
    axes[0, 1].imshow(ensemble_data[time_idx], cmap=cmap, vmin=0.0, vmax=70.0)
    mae = np.abs(ensemble_data[time_idx] - GT_data[time_idx]).mean()
    csi2 = CSI(GT_data[time_idx], ensemble_data[time_idx], 2.0)
    csi8 = CSI(GT_data[time_idx], ensemble_data[time_idx], 8.0)
    axes[0, 1].set_title(f"Ensemble Mean\nMAE={mae:.2f}, CSI2={csi2:.2f}, CSI8={csi8:.2f}", fontsize=12)
    
    # 绘制最后一帧输入
    axes[0, 2].imshow(input_data[-1], cmap=cmap, vmin=0.0, vmax=70.0)
    axes[0, 2].set_title("Last Input Frame", fontsize=12)
    
    # 第二行: 集合标准差、集合范围和概率预报
    # 绘制集合标准差
    im_std = axes[1, 0].imshow(ensemble_std, cmap='plasma', vmin=0.0)
    axes[1, 0].set_title(f"Ensemble Std Dev\nMean={ensemble_std.mean():.2f}, Max={ensemble_std.max():.2f}", fontsize=12)
    plt.colorbar(im_std, ax=axes[1, 0])
    
    # 绘制集合范围
    im_range = axes[1, 1].imshow(ensemble_range, cmap='viridis', vmin=0.0)
    axes[1, 1].set_title(f"Ensemble Range\nMean={ensemble_range.mean():.2f}, Max={ensemble_range.max():.2f}", fontsize=12)
    plt.colorbar(im_range, ax=axes[1, 1])
    
    # 绘制概率预报 (大于8mm的概率)
    threshold = 8.0
    prob_map = np.mean(all_members_stack[:, time_idx] > threshold, axis=0) * 100  # 百分比
    im_prob = axes[1, 2].imshow(prob_map, cmap='RdYlBu_r', vmin=0, vmax=100)
    axes[1, 2].set_title(f"Probability > {threshold}mm (%)", fontsize=12)
    plt.colorbar(im_prob, ax=axes[1, 2])
    
    # 格式化所有子图
    for ax_row in axes:
        for ax in ax_row:
            # 隐藏轴线
            for spine in ax.spines.values():
                spine.set_visible(False)
            # 隐藏刻度标签和刻度线
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                         labelbottom=False, labelleft=False)
            # 添加网格线
            ax.grid(linestyle='-.')
    
    # 设置整体标题
    min_time = time_idx * 6
    plt.suptitle(f"Ensemble Spread Analysis at T+{min_time}min - {file_name}", fontsize=16)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    spread_dir = os.path.join(os.path.dirname(save_dir), "ensemble_spread")
    os.makedirs(spread_dir, exist_ok=True)
    plt.savefig(f"{spread_dir}/{file_name}_T{min_time}min_spread.png", dpi=150, bbox_inches='tight')
    plt.close()