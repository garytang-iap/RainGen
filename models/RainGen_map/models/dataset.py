# file: dataset_final.py (恢复Z-Score版)

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
import warnings
'''
class RadarPrecipitationFixedRegionDataset(Dataset):
    def __init__(self, 
                 file_list,
                 normalize=True,
                 **kwargs):
        super().__init__()
        self.data_paths = file_list
        self.normalize = normalize
        
        # 固定区域的格点索引
        #self.FIXED_LAT_START = 33   
        #self.FIXED_LAT_END = 97     
        #self.FIXED_LON_START = 191  
        #self.FIXED_LON_END = 255    
        self.FIXED_LAT_START = 106  # 对应实际纬度约 12.2 + 106 * 0.2 = 33.4 N
        self.FIXED_LAT_END = 170    # 对应实际纬度约 12.2 + 170 * 0.2 = 46.2 N (确保 end - start = 64)
        self.FIXED_LON_START = 184  # 对应实际经度约 73.0 + 184 * 0.2006 = 109.8 E
        self.FIXED_LON_END = 248    # 对应实际经度约 73.0 + 248 * 0.2006 = 122.8 E (确保 end - start = 64)
        # 🔥 全国经纬度范围
        self.FULL_LON_RANGE = (73.0, 135.0)
        self.FULL_LAT_RANGE = (12.2, 54.2)
        self.FULL_GRID_SHAPE = (211, 311)
        
        # 🔥 计算固定区域的真实经纬度
        full_lons = np.linspace(self.FULL_LON_RANGE[0], self.FULL_LON_RANGE[1], self.FULL_GRID_SHAPE[1])
        full_lats = np.linspace(self.FULL_LAT_RANGE[0], self.FULL_LAT_RANGE[1], self.FULL_GRID_SHAPE[0])
        
        # 固定区域的实际经纬度
        region_lons = full_lons[self.FIXED_LON_START:self.FIXED_LON_END]  # 64个点的实际经度
        region_lats = full_lats[self.FIXED_LAT_START:self.FIXED_LAT_END]  # 64个点的实际纬度
        
        # 🔥 基于全国范围归一化到[-1, 1]
        region_lons_norm = 2 * (region_lons - self.FULL_LON_RANGE[0]) / (self.FULL_LON_RANGE[1] - self.FULL_LON_RANGE[0]) - 1
        region_lats_norm = 2 * (region_lats - self.FULL_LAT_RANGE[0]) / (self.FULL_LAT_RANGE[1] - self.FULL_LAT_RANGE[0]) - 1
        
        # 创建网格
        self.region_lon_grid, self.region_lat_grid = np.meshgrid(region_lons_norm, region_lats_norm)
        
        print(f"✅ Fixed region grid indices: lat[{self.FIXED_LAT_START}:{self.FIXED_LAT_END}], lon[{self.FIXED_LON_START}:{self.FIXED_LON_END}]")
        print(f"✅ Real lon range: [{region_lons[0]:.2f}, {region_lons[-1]:.2f}]")
        print(f"✅ Real lat range: [{region_lats[0]:.2f}, {region_lats[-1]:.2f}]")
        print(f"✅ Normalized lon range: [{region_lons_norm[0]:.3f}, {region_lons_norm[-1]:.3f}]")
        print(f"✅ Normalized lat range: [{region_lats_norm[0]:.3f}, {region_lats_norm[-1]:.3f}]")
        
        # 归一化参数
        self.radar_mean = kwargs.get('radar_mean', 0.0)
        self.radar_std = kwargs.get('radar_std', 1.0)
        self.precip_mean = kwargs.get('precip_mean', 0.0)
        self.precip_std = kwargs.get('precip_std', 1.0)
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        try:
            # 加载全国数据
            data = np.load(self.data_paths[idx])
            radar_full_log1p = data['image'].astype(np.float32)     # [5, 211, 311]
            precip_full_log1p = data['label'].astype(np.float32)    # [211, 311]
            
            # 🔥 用显式格点索引裁剪到固定64×64区域
            radar_region = radar_full_log1p[:, 
                                           self.FIXED_LAT_START:self.FIXED_LAT_END,
                                           self.FIXED_LON_START:self.FIXED_LON_END]  # [5, 64, 64]
            
            precip_region = precip_full_log1p[self.FIXED_LAT_START:self.FIXED_LAT_END,
                                            self.FIXED_LON_START:self.FIXED_LON_END]   # [64, 64]
            
            # 添加维度
            precip_region = precip_region[np.newaxis, ...]  # [1, 64, 64]
            
            # 坐标网格
            coords_region = np.stack([self.region_lon_grid, self.region_lat_grid], axis=0).astype(np.float32)  # [2, 64, 64]
            
            # 归一化
            if self.normalize:
                radar_norm = (radar_region - self.radar_mean) / self.radar_std
                precip_norm = (precip_region - self.precip_mean) / self.precip_std
            else:
                radar_norm = radar_region
                precip_norm = precip_region
            
            # 合并 [1+5, 64, 64] = [6, 64, 64]
            combined_patch = np.concatenate([precip_norm, radar_norm], axis=0)
            
            return {
                "combined_patch": torch.from_numpy(combined_patch),
                "coords_patch": torch.from_numpy(coords_region),
            }
            
        except Exception as e:
            warnings.warn(f"Error loading {self.data_paths[idx]}: {e}")
            return None
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.default_collate(batch)

'''
class RadarPrecipitationFixedRegionDataset(Dataset):
    """
    一个从全国雷达和降水数据中随机裁剪patch的数据集。

    - 在每次调用 __getitem__ 时，都会从完整的全国网格中随机选择一个
      指定大小 (crop_size) 的区域进行裁剪。
    - 坐标归一化始终基于全国的经纬度范围，以保持位置信息的一致性。
    """
    def __init__(self, 
                 file_list,
                 crop_size=128,
                 normalize=True,
                 **kwargs):
        """
        Args:
            file_list (list): 包含 .npz 文件路径的列表。
            crop_size (int): 随机裁剪的正方形区域的边长。
            normalize (bool): 是否对雷达和降水数据进行Z-Score归一化。
            **kwargs: 包含归一化参数的字典 (e.g., radar_mean, radar_std, ...)。
        """
        super().__init__()
        self.data_paths = file_list
        self.crop_size = crop_size
        self.normalize = normalize
        
        # 定义全国范围的常量
        self.FULL_LON_RANGE = (73.0, 135.0)
        self.FULL_LAT_RANGE = (12.2, 54.2)
        self.FULL_GRID_SHAPE = (211, 311) # (纬度格点数, 经度格点数)
        
        # 预先计算全国的经纬度格点，以提高效率
        self.full_lons = np.linspace(self.FULL_LON_RANGE[0], self.FULL_LON_RANGE[1], self.FULL_GRID_SHAPE[1])
        self.full_lats = np.linspace(self.FULL_LAT_RANGE[0], self.FULL_LAT_RANGE[1], self.FULL_GRID_SHAPE[0])
        
        print(f"✅ Dataset initialized for random cropping with crop_size = {crop_size}x{crop_size}.")
        print(f"✅ Coordinate normalization will be based on the full national grid.")
        
        # 存储归一化参数
        self.radar_mean = kwargs.get('radar_mean', 0.0)
        self.radar_std = kwargs.get('radar_std', 1.0)
        self.precip_mean = kwargs.get('precip_mean', 0.0)
        self.precip_std = kwargs.get('precip_std', 1.0)
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        try:
            # 1. 加载完整的全国数据
            data = np.load(self.data_paths[idx])
            radar_full_log1p = data['image'].astype(np.float32)     # [5, 211, 311]
            precip_full_log1p = data['label'].astype(np.float32)    # [211, 311]
            
            # 2. 🔥 随机确定裁剪区域的左上角索引
            max_lat_start = self.FULL_GRID_SHAPE[0] - self.crop_size
            max_lon_start = self.FULL_GRID_SHAPE[1] - self.crop_size
            
            start_lat_idx = random.randint(0, max_lat_start)
            start_lon_idx = random.randint(0, max_lon_start)
            
            end_lat_idx = start_lat_idx + self.crop_size
            end_lon_idx = start_lon_idx + self.crop_size
            
            # 3. 🔥 根据随机索引裁剪雷达和降水数据
            radar_crop = radar_full_log1p[:, start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx]
            precip_crop = precip_full_log1p[start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx]
            
            # 4. 为裁剪区域生成两套坐标

            # ---- A) coords_abs: 绝对经纬度（全国归一化到 [-1,1]）
            crop_lons = self.full_lons[start_lon_idx:end_lon_idx]
            crop_lats = self.full_lats[start_lat_idx:end_lat_idx]

            crop_lons_abs = 2 * (crop_lons - self.FULL_LON_RANGE[0]) / (self.FULL_LON_RANGE[1] - self.FULL_LON_RANGE[0]) - 1
            crop_lats_abs = 2 * (crop_lats - self.FULL_LAT_RANGE[0]) / (self.FULL_LAT_RANGE[1] - self.FULL_LAT_RANGE[0]) - 1

            lon_abs_grid, lat_abs_grid = np.meshgrid(crop_lons_abs, crop_lats_abs)
            coords_abs = np.stack([lon_abs_grid, lat_abs_grid], axis=0).astype(np.float32)  # [2,H,W]

            # ---- B) coords_geo: 几何坐标（patch内规则坐标，覆盖满 [-1,1]）
            geo_x = np.linspace(-1, 1, self.crop_size, dtype=np.float32)
            geo_y = np.linspace(-1, 1, self.crop_size, dtype=np.float32)
            gx, gy = np.meshgrid(geo_x, geo_y)
            coords_geo = np.stack([gx, gy], axis=0).astype(np.float32)  # [2,H,W]


            # 5. 后续处理 (与之前相同)
            # 添加通道维度
            precip_crop = precip_crop[np.newaxis, ...]  # [1, 64, 64]
            
            # Z-Score 归一化
            if self.normalize:
                radar_norm = (radar_crop - self.radar_mean) / self.radar_std
                precip_norm = (precip_crop - self.precip_mean) / self.precip_std
            else:
                radar_norm = radar_crop
                precip_norm = precip_crop
            
            # 合并成一个张量 [1+5, 64, 64]
            combined_patch = np.concatenate([precip_norm, radar_norm], axis=0)
            
            return {
                    "combined_patch": torch.from_numpy(combined_patch),
                    "coords_abs_patch": torch.from_numpy(coords_abs),
                    "coords_geo_patch": torch.from_numpy(coords_geo),
                }

            
        except Exception as e:
            warnings.warn(f"Error loading or processing {self.data_paths[idx]}: {e}")
            return None

def custom_collate_fn(batch):
    """
    过滤掉在 __getitem__ 中因错误而返回 None 的样本。
    """
    batch = [item for item in batch if item is not None]
    if not batch: 
        return None
    return torch.utils.data.default_collate(batch)