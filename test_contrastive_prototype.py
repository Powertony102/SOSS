#!/usr/bin/env python3
"""
测试基于SS-Net的对比学习原型管理器
验证对比学习损失计算的正确性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from myutils.contrastive_prototype_manager import ContrastivePrototypeManager

def test_contrastive_prototype_manager():
    """测试对比学习原型管理器"""
    print("🧪 测试基于SS-Net的对比学习原型管理器...")
    
    # 设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 参数
    num_classes = 3  # 背景 + 2个前景类
    feature_dim = 256  # 按照SS-Net的特征维度
    elements_per_class = 16  # 测试用较小数值
    batch_size = 2
    H, W, D = 32, 32, 32
    
    # 创建管理器 - 不使用学习的选择器版本
    print("\n📋 测试1: 简单版本（不使用学习的选择器）")
    simple_manager = ContrastivePrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        elements_per_class=elements_per_class,
        confidence_threshold=0.8,
        use_learned_selector=False,
        device=device
    )
    
    # 模拟数据
    features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
    predictions = torch.randn(batch_size, num_classes, H, W, D, device=device)
    labels = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
    
    print(f"✅ 创建模拟数据成功 - 特征形状: {features.shape}")
    
    try:
        # 第一次更新和损失计算
        total_loss, loss_dict = simple_manager.update_and_compute_loss(
            features=features,
            predictions=predictions,
            labels=labels,
            is_labeled=True,
            contrastive_weight=1.0,
            intra_weight=0.1,
            inter_weight=0.1
        )
        
        print(f"✅ 第一次损失计算成功:")
        for key, value in loss_dict.items():
            print(f"   {key}: {value:.4f}")
            
        # 检查内存状态
        memory_info = simple_manager.get_memory_info()
        print(f"✅ 内存状态: {memory_info['memory_status']}")
        
        # 第二次更新（应该有对比学习损失）
        new_features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
        new_predictions = torch.randn(batch_size, num_classes, H, W, D, device=device)
        new_labels = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
        
        total_loss2, loss_dict2 = simple_manager.update_and_compute_loss(
            features=new_features,
            predictions=new_predictions,
            labels=new_labels,
            is_labeled=True,
            contrastive_weight=1.0,
            intra_weight=0.1,
            inter_weight=0.1
        )
        
        print(f"✅ 第二次损失计算成功:")
        for key, value in loss_dict2.items():
            print(f"   {key}: {value:.4f}")
            
        # 检查梯度
        if total_loss2.requires_grad:
            total_loss2.backward()
            if new_features.grad is not None:
                grad_norm = torch.norm(new_features.grad).item()
                print(f"✅ 梯度反向传播成功 - 梯度范数: {grad_norm:.4f}")
            else:
                print("❌ 未检测到梯度")
                return False
        
    except Exception as e:
        print(f"❌ 简单版本测试失败: {e}")
        return False
    
    # 测试学习的选择器版本
    print("\n📋 测试2: 学习的选择器版本")
    learned_manager = ContrastivePrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        elements_per_class=elements_per_class,
        confidence_threshold=0.8,
        use_learned_selector=True,
        device=device
    )
    
    try:
        # 重置梯度
        if features.grad is not None:
            features.grad.zero_()
        
        # 使用学习的选择器计算损失
        features_learned = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
        predictions_learned = torch.randn(batch_size, num_classes, H, W, D, device=device)
        labels_learned = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
        
        # 第一次更新
        loss1, loss_dict1 = learned_manager.update_and_compute_loss(
            features=features_learned,
            predictions=predictions_learned,
            labels=labels_learned,
            is_labeled=True
        )
        
        # 第二次更新（应该有对比学习损失）
        features_learned2 = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
        predictions_learned2 = torch.randn(batch_size, num_classes, H, W, D, device=device)
        labels_learned2 = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
        
        loss2, loss_dict2 = learned_manager.update_and_compute_loss(
            features=features_learned2,
            predictions=predictions_learned2,
            labels=labels_learned2,
            is_labeled=True
        )
        
        print(f"✅ 学习的选择器版本损失计算成功:")
        for key, value in loss_dict2.items():
            print(f"   {key}: {value:.4f}")
            
        # 检查选择器模块
        feature_selectors, memory_selectors = learned_manager.get_selectors()
        if feature_selectors is not None and memory_selectors is not None:
            print(f"✅ 选择器模块数量: 特征选择器={len(feature_selectors)}, 内存选择器={len(memory_selectors)}")
        
        # 梯度检查
        loss2.backward()
        if features_learned2.grad is not None:
            grad_norm = torch.norm(features_learned2.grad).item()
            print(f"✅ 学习版本梯度反向传播成功 - 梯度范数: {grad_norm:.4f}")
        
    except Exception as e:
        print(f"❌ 学习的选择器版本测试失败: {e}")
        return False
    
    print("\n📋 测试3: 无标签数据测试")
    try:
        # 无标签数据测试
        unlabeled_features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
        unlabeled_predictions = torch.randn(batch_size, num_classes, H, W, D, device=device)
        
        # 创建高置信度预测
        unlabeled_predictions[:, 1, :, :, :] = 5.0  # 高置信度预测为类别1
        
        loss_unlabeled, loss_dict_unlabeled = simple_manager.update_and_compute_loss(
            features=unlabeled_features,
            predictions=unlabeled_predictions,
            labels=None,
            is_labeled=False
        )
        
        print(f"✅ 无标签数据测试成功:")
        for key, value in loss_dict_unlabeled.items():
            print(f"   {key}: {value:.4f}")
            
    except Exception as e:
        print(f"❌ 无标签数据测试失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！对比学习原型管理器工作正常。")
    return True

def test_contrastive_loss_computation():
    """单独测试对比学习损失计算"""
    print("\n🔍 详细测试对比学习损失计算...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 2  # 简化为2类
    feature_dim = 64
    
    # 创建管理器
    manager = ContrastivePrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        elements_per_class=4,  # 小数量便于观察
        confidence_threshold=0.5,
        use_learned_selector=False,
        device=device
    )
    
    # 手动创建内存
    for class_id in range(num_classes):
        # 为每个类创建不同的特征分布
        class_features = torch.randn(4, feature_dim, device=device) + class_id * 2.0
        manager.memory[class_id] = class_features.cpu().numpy()
    
    manager.initialized = True
    
    # 创建测试特征
    test_features = torch.randn(8, feature_dim, device=device, requires_grad=True)
    test_labels = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0], device=device)
    
    # 计算对比学习损失
    contrastive_loss = manager.contrastive_class_to_class_learned_memory(test_features, test_labels)
    
    print(f"✅ 对比学习损失: {contrastive_loss.item():.4f}")
    
    # 检查损失的合理性
    if contrastive_loss.item() > 0:
        print("✅ 损失值合理（大于0）")
    else:
        print("⚠️ 损失值为0，可能需要检查")
    
    # 梯度检查
    contrastive_loss.backward()
    if test_features.grad is not None:
        grad_norm = torch.norm(test_features.grad).item()
        print(f"✅ 梯度计算成功 - 梯度范数: {grad_norm:.4f}")
    
    return True

def main():
    """主函数"""
    print("🚀 基于SS-Net的对比学习原型管理器测试套件")
    print("=" * 60)
    
    # 运行基本功能测试
    success1 = test_contrastive_prototype_manager()
    
    if success1:
        # 运行详细的损失计算测试
        success2 = test_contrastive_loss_computation()
        
        if success2:
            print("\n💡 实现要点:")
            print("1. 严格按照SS-Net的对比学习损失计算方式")
            print("2. 使用L2归一化和相似性矩阵计算")
            print("3. 支持学习的特征选择器（可选）")
            print("4. 正确的梯度管理和内存更新")
            print("5. 与原有框架兼容的接口设计")
            
            print("\n🎯 使用建议:")
            print("- 对于初期训练，使用简单版本 (use_learned_selector=False)")
            print("- 对于高级应用，可以启用学习的选择器")
            print("- 调整 contrastive_weight 来平衡对比学习和传统损失")
            print("- 监控对比学习损失的变化趋势")
    
    else:
        print("\n❌ 测试失败，请检查实现")

if __name__ == "__main__":
    main() 