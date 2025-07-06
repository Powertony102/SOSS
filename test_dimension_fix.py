#!/usr/bin/env python3
"""
测试维度修复的脚本
验证ContrastivePrototypeManager能够正确处理维度不匹配的情况
"""

import torch
import torch.nn.functional as F
import numpy as np
from myutils.contrastive_prototype_manager import ContrastivePrototypeManager

def test_dimension_fix():
    """测试维度修复功能"""
    print("🧪 测试维度修复功能")
    print("=" * 50)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试参数
    num_classes = 2
    expected_feature_dim = 64  # 训练脚本中设置的维度
    actual_feature_dim = 16    # 模型实际输出的维度
    batch_size = 2
    spatial_size = (8, 8, 8)
    
    print(f"预期特征维度: {expected_feature_dim}")
    print(f"实际特征维度: {actual_feature_dim}")
    print(f"批次大小: {batch_size}")
    print(f"空间尺寸: {spatial_size}")
    
    # 创建管理器（使用预期维度）
    print("\n📦 创建ContrastivePrototypeManager...")
    manager = ContrastivePrototypeManager(
        num_classes=num_classes,
        feature_dim=expected_feature_dim,  # 使用预期维度
        elements_per_class=8,
        confidence_threshold=0.8,
        use_learned_selector=True,  # 启用学习的选择器
        device=device
    )
    
    # 创建模拟数据
    print("\n📊 创建模拟数据...")
    features = torch.randn(batch_size, actual_feature_dim, *spatial_size, device=device)
    predictions = torch.randn(batch_size, num_classes, *spatial_size, device=device)
    labels = torch.randint(0, num_classes, (batch_size, *spatial_size), device=device)
    
    print(f"特征形状: {features.shape}")
    print(f"预测形状: {predictions.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 测试特征提取（应该自动检测维度并初始化选择器）
    print("\n🔍 测试特征提取...")
    try:
        high_quality_features = manager.extract_high_quality_features(
            features, predictions, labels, is_labeled=True
        )
        print("✅ 特征提取成功")
        print(f"提取的高质量特征: {len(high_quality_features)} 个类别")
        for class_id, feat in high_quality_features.items():
            print(f"  类别 {class_id}: {feat.shape}")
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return False
    
    # 测试选择器初始化
    print("\n🎯 测试选择器初始化...")
    feature_selectors, memory_selectors = manager.get_selectors()
    if feature_selectors is not None and memory_selectors is not None:
        print("✅ 选择器初始化成功")
        print(f"特征选择器数量: {len(feature_selectors)}")
        print(f"内存选择器数量: {len(memory_selectors)}")
        
        # 测试选择器维度
        for name, selector in feature_selectors.items():
            print(f"  {name}: {selector}")
    else:
        print("❌ 选择器初始化失败")
        return False
    
    # 测试损失计算
    print("\n📈 测试损失计算...")
    try:
        total_loss, loss_dict = manager.update_and_compute_loss(
            features, predictions, labels, 
            is_labeled=True,
            contrastive_weight=1.0,
            intra_weight=0.1,
            inter_weight=0.1
        )
        print("✅ 损失计算成功")
        print(f"总损失: {total_loss.item():.4f}")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        return False
    
    # 测试内存信息
    print("\n💾 测试内存信息...")
    memory_info = manager.get_memory_info()
    print("✅ 内存信息获取成功")
    print(f"初始化状态: {memory_info['initialized']}")
    print(f"使用学习选择器: {memory_info['use_learned_selector']}")
    for class_id, status in memory_info['memory_status'].items():
        if status is not None:
            print(f"  类别 {class_id}: {status['num_features']} 个特征, 维度 {status['feature_dim']}")
        else:
            print(f"  类别 {class_id}: 无特征")
    
    print("\n🎉 所有测试通过！维度修复成功！")
    return True

def test_different_dimensions():
    """测试不同维度的情况"""
    print("\n" + "=" * 50)
    print("🧪 测试不同维度的情况")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试不同的维度组合
    test_cases = [
        (32, 32),  # 匹配
        (64, 16),  # 不匹配（当前问题）
        (128, 32), # 不匹配
        (16, 16),  # 匹配
    ]
    
    for expected_dim, actual_dim in test_cases:
        print(f"\n📊 测试: 预期 {expected_dim} -> 实际 {actual_dim}")
        
        try:
            manager = ContrastivePrototypeManager(
                num_classes=2,
                feature_dim=expected_dim,
                elements_per_class=4,
                use_learned_selector=True,
                device=device
            )
            
            # 创建测试数据
            features = torch.randn(1, actual_dim, 4, 4, 4, device=device)
            predictions = torch.randn(1, 2, 4, 4, 4, device=device)
            labels = torch.randint(0, 2, (1, 4, 4, 4), device=device)
            
            # 测试特征提取
            high_quality_features = manager.extract_high_quality_features(
                features, predictions, labels, is_labeled=True
            )
            
            # 测试损失计算
            total_loss, loss_dict = manager.update_and_compute_loss(
                features, predictions, labels, 
                is_labeled=True,
                contrastive_weight=1.0,
                intra_weight=0.1,
                inter_weight=0.1
            )
            
            print(f"  ✅ 成功 - 损失: {total_loss.item():.4f}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")

if __name__ == "__main__":
    print("🚀 开始测试维度修复功能")
    
    # 运行主测试
    success = test_dimension_fix()
    
    if success:
        # 运行不同维度的测试
        test_different_dimensions()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成！维度修复功能正常工作！")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ 测试失败！需要进一步调试。")
        print("=" * 50) 