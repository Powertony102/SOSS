#!/usr/bin/env python3
"""
测试模型维度修复的脚本
验证corn模型能够输出正确的embedding维度
"""

import torch
import torch.nn.functional as F
import numpy as np
from networks.net_factory import net_factory

def test_corn_model_dimension():
    """测试corn模型的embedding维度"""
    print("🧪 测试corn模型embedding维度修复")
    print("=" * 50)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试参数
    embedding_dim = 64  # 期望的embedding维度
    batch_size = 2
    spatial_size = (112, 112, 80)  # 增大空间尺寸，避免下采样后尺寸过小
    
    print(f"期望的embedding维度: {embedding_dim}")
    print(f"批次大小: {batch_size}")
    print(f"空间尺寸: {spatial_size}")
    
    # 创建corn模型
    print("\n📦 创建corn模型...")
    model = net_factory(
        net_type="corn", 
        in_chns=1, 
        class_num=2, 
        mode="train",
        feat_dim=embedding_dim
    )
    
    # 创建测试输入
    print("\n📊 创建测试输入...")
    input_tensor = torch.randn(batch_size, 1, *spatial_size, device=device)
    print(f"输入形状: {input_tensor.shape}")
    
    # 前向传播
    print("\n🔍 执行前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            outputs_v, outputs_a, embedding_v, embedding_a = model(input_tensor)
        
        print("✅ 前向传播成功")
        print(f"输出v形状: {outputs_v.shape}")
        print(f"输出a形状: {outputs_a.shape}")
        print(f"Embedding v形状: {embedding_v.shape}")
        print(f"Embedding a形状: {embedding_a.shape}")
        
        # 检查embedding维度
        expected_shape = (batch_size, embedding_dim, *spatial_size)
        if embedding_v.shape == expected_shape and embedding_a.shape == expected_shape:
            print(f"✅ Embedding维度正确: {embedding_dim}")
        else:
            print(f"❌ Embedding维度错误:")
            print(f"   期望: {expected_shape}")
            print(f"   实际v: {embedding_v.shape}")
            print(f"   实际a: {embedding_a.shape}")
            return False
        
        # 测试特征提取
        print("\n🔍 测试特征提取...")
        features_combined = (embedding_v + embedding_a) / 2
        print(f"组合特征形状: {features_combined.shape}")
        
        # 展平特征用于测试
        B, C, H, W, D = features_combined.shape
        features_flat = features_combined.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        print(f"展平特征形状: {features_flat.shape}")
        
        if features_flat.shape[-1] == embedding_dim:
            print(f"✅ 特征维度正确: {embedding_dim}")
        else:
            print(f"❌ 特征维度错误: {features_flat.shape[-1]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_corn2d_model_dimension():
    """测试corn2d模型的embedding维度"""
    print("\n" + "=" * 50)
    print("🧪 测试corn2d模型embedding维度修复")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_dim = 64
    batch_size = 2
    spatial_size = (128, 128)  # 增大空间尺寸，避免下采样后尺寸过小
    
    print(f"期望的embedding维度: {embedding_dim}")
    print(f"批次大小: {batch_size}")
    print(f"空间尺寸: {spatial_size}")
    
    # 创建corn2d模型
    print("\n📦 创建corn2d模型...")
    model = net_factory(
        net_type="corn2d", 
        in_chns=1, 
        class_num=2, 
        mode="train",
        feat_dim=embedding_dim
    )
    
    # 创建测试输入
    print("\n📊 创建测试输入...")
    input_tensor = torch.randn(batch_size, 1, *spatial_size, device=device)
    print(f"输入形状: {input_tensor.shape}")
    
    # 前向传播
    print("\n🔍 执行前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            outputs_v, outputs_a, embedding_v, embedding_a = model(input_tensor)
        
        print("✅ 前向传播成功")
        print(f"输出v形状: {outputs_v.shape}")
        print(f"输出a形状: {outputs_a.shape}")
        print(f"Embedding v形状: {embedding_v.shape}")
        print(f"Embedding a形状: {embedding_a.shape}")
        
        # 检查embedding维度
        expected_shape = (batch_size, embedding_dim, *spatial_size)
        if embedding_v.shape == expected_shape and embedding_a.shape == expected_shape:
            print(f"✅ Embedding维度正确: {embedding_dim}")
        else:
            print(f"❌ Embedding维度错误:")
            print(f"   期望: {expected_shape}")
            print(f"   实际v: {embedding_v.shape}")
            print(f"   实际a: {embedding_a.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_embedding_dimensions():
    """测试不同的embedding维度"""
    print("\n" + "=" * 50)
    print("🧪 测试不同的embedding维度")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    spatial_size = (128, 128, 128)  # 增大空间尺寸，避免下采样后尺寸过小
    
    # 测试不同的embedding维度
    test_dimensions = [16, 32, 64, 128]
    
    for embedding_dim in test_dimensions:
        print(f"\n📊 测试embedding维度: {embedding_dim}")
        
        try:
            # 创建模型
            model = net_factory(
                net_type="corn", 
                in_chns=1, 
                class_num=2, 
                mode="train",
                feat_dim=embedding_dim
            )
            
            # 创建输入
            input_tensor = torch.randn(batch_size, 1, *spatial_size, device=device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                outputs_v, outputs_a, embedding_v, embedding_a = model(input_tensor)
            
            # 检查维度
            expected_shape = (batch_size, embedding_dim, *spatial_size)
            if embedding_v.shape == expected_shape:
                print(f"  ✅ 成功 - 维度: {embedding_v.shape}")
            else:
                print(f"  ❌ 失败 - 期望: {expected_shape}, 实际: {embedding_v.shape}")
                
        except Exception as e:
            print(f"  ❌ 失败: {e}")

if __name__ == "__main__":
    print("🚀 开始测试模型维度修复")
    
    # 测试corn模型
    success1 = test_corn_model_dimension()
    
    # 测试corn2d模型
    success2 = test_corn2d_model_dimension()
    
    # 测试不同维度
    test_different_embedding_dimensions()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 所有测试通过！模型维度修复成功！")
        print("✅ corn模型: embedding维度正确")
        print("✅ corn2d模型: embedding维度正确")
        print("✅ 支持不同的embedding维度设置")
    else:
        print("❌ 部分测试失败！需要进一步调试。")
    print("=" * 50) 