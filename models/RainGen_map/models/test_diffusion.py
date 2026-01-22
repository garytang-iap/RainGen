# file: your_project/test_diffusion_training.py

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import math

# =============================================================================
# 准备工作: 导入所有需要的模块
# 在你的实际项目中，这些应该使用正确的相对/绝对路径导入
# =============================================================================

# 假设这些是你已经修改好的文件
from diffusion import GaussianDiffusion, ModelMeanType, LossType, get_named_beta_schedule,ModelVarType
from model_self import ConditionalSparseUNet # 你的模型包装器

def test_diffusion_training_step():
    """
    测试改造后的 GaussianDiffusion 的 training_losses 方法是否正常工作。
    模拟一个完整的 "网格 -> 散点 -> 模型 -> 损失" 流程。
    """
    print("🚀 Starting Diffusion Training Step Test...")

    # 1. 定义测试参数
    batch_size = 4
    precip_channels = 1
    radar_channels = 5
    crop_size = 64  # 使用一个较小的尺寸以加快测试
    num_points = 1024 # 采样点数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Parameters: Batch={batch_size}, Crop Size={crop_size}x{crop_size}, Points={num_points}, Device={device}")

    # 2. 实例化模型和扩散过程
    # (a) 实例化模型
    # 注意：这里的 img_size 应该与训练时的 patch size 一致
    model = ConditionalSparseUNet(
        radar_channels=radar_channels,
        precip_channels=precip_channels,
        img_size=crop_size,
        # ... 其他 SparseUNet 的必要参数
        nf=32, uno_res=16, time_emb_dim=128, backend="torch_dense"
    ).to(device)
    model.eval() # 设为评估模式进行测试
    print("✓ ConditionalSparseUNet model instantiated.")

    # (b) 实例化扩散过程
    betas = get_named_beta_schedule("cosine", 1000)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.MOLLIFIED_EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE, # 使用一个固定类型
        loss_type=LossType.MSE,
        gaussian_filter_std=1.0, # 开启 Mollification
        img_size=crop_size
    ).to(device)
    print("✓ GaussianDiffusion process instantiated.")

    # 3. 创建假的输入数据
    # (a) 创建完整的网格数据
    precip_grid = torch.randn(batch_size, precip_channels, crop_size, crop_size, device=device)
    condition_grid = torch.randn(batch_size, radar_channels, crop_size, crop_size, device=device)
    
    # (b) 动态生成散点信息
    sample_lst = torch.stack([
        torch.from_numpy(np.random.choice(crop_size**2, num_points, replace=False))
        for _ in range(batch_size)
    ]).to(device)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, crop_size, device=device),
        torch.linspace(-1, 1, crop_size, device=device),
        indexing='ij'
    )
    coords_grid_flat = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
    # 使用 _gather_from_grid 来采样坐标，保持逻辑一致性
    coords_grid = rearrange(coords_grid_flat, '(h w) c -> () c h w', h=crop_size, w=crop_size)
    coords_points = diffusion._gather_from_grid(coords_grid.repeat(batch_size,1,1,1), sample_lst)

    # (c) 准备时间步
    t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device).long()
    
    # (d) 打包 model_kwargs
    model_kwargs = {
        "condition_grid": condition_grid,
        "sample_lst": sample_lst,
        "coords": coords_points
    }
    
    print("✓ Fake grid and point data created successfully.")

    # 4. 执行 training_losses
    try:
        print("\n🔍 Running diffusion.training_losses()...")
        # 调用我们重写过的函数
        loss_dict = diffusion.training_losses(
            model=model,
            x_start=precip_grid,
            t=t,
            model_kwargs=model_kwargs
        )
        print("✓ training_losses completed without errors.")
    except Exception as e:
        print(f"❌ Test Failed during training_losses call!")
        raise e

    # 5. 验证输出
    print("\n🧐 Verifying output...")
    assert "loss" in loss_dict, "Output dictionary must contain a 'loss' key."
    print("✓ 'loss' key found in output dictionary.")
    
    loss = loss_dict["loss"]
    expected_shape = (batch_size,)
    assert loss.shape == expected_shape, \
        f"Loss shape is incorrect! Expected {expected_shape}, but got {loss.shape}"
    print(f"✓ Loss shape is correct: {loss.shape}")
    
    assert loss.dtype == torch.float32, \
        f"Loss dtype is incorrect! Expected torch.float32, but got {loss.dtype}"
    print(f"✓ Loss dtype is correct: {loss.dtype}")

    # 检查损失值是否有效
    assert not torch.any(torch.isnan(loss)) and not torch.any(torch.isinf(loss)), \
        "Loss contains NaN or Inf values."
    print("✓ Loss values are finite.")

    print("\n✅✅✅ All tests passed! The modified diffusion process works correctly with the conditional sparse model.")

if __name__ == '__main__':
    # 为了让这个脚本独立运行，你需要：
    # 1. 确保 lib/diffusion.py 已经被我们的最终版本替换。
    # 2. 确保 model.py 中有 ConditionalSparseUNet。
    # 3. 确保所有依赖（如 sparse_unet.py, conv_uno.py）都在正确的路径下。
    test_diffusion_training_step()