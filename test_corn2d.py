#!/usr/bin/env python3

import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from networks.net_factory import net_factory

def test_corn2d():
    """测试corn2d模型是否能正确处理2D数据"""
    print("测试corn2d模型...")
    
    # 创建模型
    try:
        model = net_factory(net_type="corn2d", in_chns=1, class_num=4, mode="train")
        print("✓ corn2d模型创建成功")
    except Exception as e:
        print(f"✗ corn2d模型创建失败: {e}")
        return False
    
    # 创建测试数据 - 模拟ACDC 2D数据
    batch_size = 2
    channels = 1
    height = 256
    width = 256
    
    test_input = torch.randn(batch_size, channels, height, width).cuda()
    print(f"✓ 测试输入创建成功: {test_input.shape}")
    
    # 测试前向传播
    try:
        model.eval()
        with torch.no_grad():
            # 测试基本前向传播
            output = model(test_input)
            print(f"✓ 基本前向传播成功: {len(output)} outputs")
            print(f"  - Output 1 shape: {output[0].shape}")
            print(f"  - Output 2 shape: {output[1].shape}")
            
            # 测试带HCC的前向传播
            output_hcc = model(test_input, with_hcc=True)
            print(f"✓ HCC前向传播成功: {len(output_hcc)} outputs")
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    print("✓ 所有测试通过！corn2d模型可以正确处理2D数据")
    return True

if __name__ == "__main__":
    success = test_corn2d()
    if success:
        print("\n🎉 corn2d模型测试成功！可以开始训练ACDC数据集")
    else:
        print("\n❌ corn2d模型测试失败，需要进一步调试")
    
    sys.exit(0 if success else 1) 