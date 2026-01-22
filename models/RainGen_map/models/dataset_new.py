# file: dataset_patched.py

import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset


class RadarPrecipitationPatchedDataset(Dataset):
    """
    用于加载预处理后的patch数据的Dataset类
    
    预处理后的数据格式：
    - image: [5, crop_size, crop_size] 雷达数据
    - label: [crop_size, crop_size] 降水数据  
    - coords: [2, crop_size, crop_size] 坐标数据
    """
    def __init__(self,
                 file_list,
                 normalize=True,
                 # Z-Score标准化参数
                 radar_mean=0.525924,
                 radar_std=0.989234,
                 precip_mean=0.104368,
                 precip_std=0.336201,
                 **kwargs
                ):
        super().__init__()
        self.data_paths = file_list
        if not self.data_paths:
            warnings.warn("Dataset initialized with zero files.")
        
        self.normalize = normalize
        # Z-Score参数
        self.radar_mean = radar_mean
        self.radar_std = radar_std
        self.precip_mean = precip_mean
        self.precip_std = precip_std
        
        print(f"📊 PatchedDataset initialized with {len(self.data_paths)} files")
        print(f"🔧 Normalization: {self.normalize}")
        if self.normalize:
            print(f"   Radar - Mean: {self.radar_mean:.6f}, Std: {self.radar_std:.6f}")
            print(f"   Precip - Mean: {self.precip_mean:.6f}, Std: {self.precip_std:.6f}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        try:
            # 加载预处理后的patch数据
            data = np.load(self.data_paths[idx])
            
            # 数据格式：
            # - image: [5, crop_size, crop_size] 雷达数据
            # - label: [crop_size, crop_size] 降水数据
            # - coords: [2, crop_size, crop_size] 坐标数据
            radar_patch_log1p = data['image'].astype(np.float32)  # [5, H, W]
            precip_patch_log1p = data['label'].astype(np.float32)  # [H, W]
            coords_patch = data['coords'].astype(np.float32)  # [2, H, W]
            
            # 确保降水数据是3D的 [1, H, W]
            if precip_patch_log1p.ndim == 2:
                precip_patch_log1p = precip_patch_log1p[np.newaxis, ...]  # [1, H, W]
            
            # Z-Score标准化
            if self.normalize:
                radar_norm = (radar_patch_log1p - self.radar_mean) / self.radar_std
                precip_norm = (precip_patch_log1p - self.precip_mean) / self.precip_std
            else:
                radar_norm = radar_patch_log1p
                precip_norm = precip_patch_log1p
            
            # 合并数据：[precip, radar] -> [1+5, H, W]
            combined_patch = np.concatenate([precip_norm, radar_norm], axis=0)
            
            return {
                "combined_patch": torch.from_numpy(combined_patch),  # [6, H, W]
                "coords_patch": torch.from_numpy(coords_patch),      # [2, H, W]
            }
            
        except Exception as e:
            warnings.warn(f"Error loading file {self.data_paths[idx]}: {e}")
            return None


def custom_collate_fn(batch):
    """自定义的collate函数，过滤掉None样本"""
    # 过滤掉None样本
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # 使用默认的collate方法
    return torch.utils.data.dataloader.default_collate(batch)


# 辅助函数：验证数据集
def verify_patched_dataset(dataset, num_samples=3):
    """验证数据集的数据格式和内容"""
    print(f"\n🔍 验证PatchedDataset (检查前{num_samples}个样本):")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            if sample is None:
                print(f"  ❌ Sample {i}: None")
                continue
                
            combined_patch = sample["combined_patch"]
            coords_patch = sample["coords_patch"]
            
            print(f"  ✅ Sample {i}:")
            print(f"     combined_patch shape: {combined_patch.shape}")
            print(f"     coords_patch shape: {coords_patch.shape}")
            
            # 检查数据范围
            precip_data = combined_patch[0]  # 第一个通道是降水
            radar_data = combined_patch[1:]  # 后面5个通道是雷达
            
            print(f"     precip range: [{precip_data.min():.3f}, {precip_data.max():.3f}]")
            print(f"     radar range: [{radar_data.min():.3f}, {radar_data.max():.3f}]")
            print(f"     coords range: [{coords_patch.min():.3f}, {coords_patch.max():.3f}]")
            
        except Exception as e:
            print(f"  ❌ Sample {i}: Error - {e}")


if __name__ == "__main__":
    # 测试代码
    import glob
    
    # 示例：测试数据集
    patch_dir = "/path/to/your/patch/directory"  # 替换为您的patch数据目录
    
    if os.path.exists(patch_dir):
        patch_files = sorted(glob.glob(os.path.join(patch_dir, "*.npz")))
        print(f"找到 {len(patch_files)} 个patch文件")
        
        if len(patch_files) > 0:
            # 创建数据集
            dataset = RadarPrecipitationPatchedDataset(
                file_list=patch_files[:100],  # 只测试前100个文件
                normalize=True
            )
            
            # 验证数据集
            verify_patched_dataset(dataset, num_samples=3)
            
            # 测试DataLoader
            from torch.utils.data import DataLoader
            
            dataloader = DataLoader(
                dataset, 
                batch_size=4, 
                shuffle=True, 
                collate_fn=custom_collate_fn,
                num_workers=2
            )
            
            print(f"\n🔍 测试DataLoader:")
            for i, batch in enumerate(dataloader):
                if batch is None:
                    print(f"  ❌ Batch {i}: None")
                    continue
                    
                print(f"  ✅ Batch {i}:")
                print(f"     combined_patch shape: {batch['combined_patch'].shape}")
                print(f"     coords_patch shape: {batch['coords_patch'].shape}")
                
                if i >= 2:  # 只测试前3个batch
                    break
    else:
        print(f"❌ 目录不存在: {patch_dir}")