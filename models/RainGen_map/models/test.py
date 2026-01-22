# file: your_project/test_model.py

import torch
import numpy as np

# 直接从你的模型文件中导入你做好的模型
from model_self import ConditionalSparseUNet 

def test_model_forward_pass():
    """
    测试ConditionalSparseUNet模型的前向传播是否正常工作。
    """
    print("🚀 Starting model forward pass test...")

    # 1. 定义测试参数
    batch_size = 4
    num_points = 1024
    radar_channels = 5
    precip_channels = 1
    img_size = 160
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Parameters: Batch={batch_size}, Points={num_points}, Device={device}")

    # 2. 实例化模型
    # 这些参数应该与你最终训练时使用的配置一致
    model = ConditionalSparseUNet(
        radar_channels=radar_channels,
        precip_channels=precip_channels,
        # 传递所有SparseUNet需要的参数
        nf=64,
        time_emb_dim=256,
        img_size=img_size,
        num_conv_blocks=3,
        knn_neighbours=3,
        uno_res=32,
        backend="torch_dense" # 关键：先用这个后端测试
    ).to(device)
    
    model.eval()
    print("✓ Model instantiated successfully.")

    # 3. 创建假的输入数据
    x_t_points = torch.randn(batch_size, num_points, precip_channels, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    condition_points = torch.randn(batch_size, num_points, radar_channels, device=device)
    coords = torch.rand(batch_size, num_points, 2, device=device) * 2 - 1
    sample_lst = torch.stack(
        [torch.from_numpy(np.random.choice(img_size**2, num_points, replace=False)) for _ in range(batch_size)]
    ).to(device)

    model_kwargs = {
        "condition_points": condition_points,
        "sample_lst": sample_lst,
        "coords": coords
    }
    
    print("✓ Fake input data created.")
    
    # 4. 执行前向传播
    try:
        with torch.no_grad():
            print("\n🔍 Running model.forward()...")
            output = model(x_t_points, t, model_kwargs)
            print("✓ Forward pass completed without errors.")
    except Exception as e:
        print(f"❌ Test Failed during forward pass!")
        raise e

    # 5. 验证输出
    print("\n🧐 Verifying output...")
    expected_shape = (batch_size, num_points, precip_channels)
    assert output.shape == expected_shape, \
        f"Output shape is incorrect! Expected {expected_shape}, but got {output.shape}"
    print(f"✓ Output shape is correct: {output.shape}")

    print("\n✅✅✅ All tests passed! The model structure and data flow seem correct.")
    return model

if __name__ == '__main__':
    # 运行测试
    test_model_forward_pass()