#!/usr/bin/env python3
"""
测试改进的原型管理器功能
验证多特征存储、在线替换和梯度管理的正确性
"""

import torch
import torch.nn as nn
import numpy as np
from myutils.improved_prototype_manager import ImprovedPrototypeManager

def test_improved_prototype_manager():
    """测试改进的原型管理器基本功能"""
    print("🧪 测试改进的原型管理器...")
    
    # 设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 参数
    num_classes = 3  # 背景 + 2个前景类
    feature_dim = 64
    elements_per_class = 8  # 测试用较小数值
    batch_size = 2
    H, W, D = 32, 32, 32
    
    # 创建管理器
    prototype_manager = ImprovedPrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        elements_per_class=elements_per_class,
        confidence_threshold=0.8,
        device=device
    )
    
    print(f"✅ 创建管理器成功 - 类别数: {num_classes}, 特征维度: {feature_dim}")
    
    # 模拟数据
    features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
    predictions = torch.randn(batch_size, num_classes, H, W, D, device=device)
    labels = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
    
    print(f"✅ 创建模拟数据成功 - 特征形状: {features.shape}")
    
    # 测试1: 初始化和特征提取
    print("\n📋 测试1: 特征提取和内存更新")
    
    try:
        # 更新特征内存并计算损失
        total_loss, loss_dict = prototype_manager.update_and_compute_loss(
            features=features,
            predictions=predictions,
            labels=labels,
            is_labeled=True,
            intra_weight=1.0,
            inter_weight=0.1,
            contrastive_weight=0.5
        )
        
        print(f"✅ 损失计算成功:")
        for key, value in loss_dict.items():
            print(f"   {key}: {value:.4f}")
        
        # 检查特征内存
        memory_info = prototype_manager.get_memory_info()
        print(f"✅ 特征内存状态: {memory_info['memory_status']}")
        
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
        return False
    
    # 测试2: 梯度检查
    print("\n📋 测试2: 梯度反向传播")
    
    try:
        # 计算损失并反向传播
        total_loss.backward()
        
        # 检查梯度
        if features.grad is not None:
            grad_norm = torch.norm(features.grad).item()
            print(f"✅ 梯度反向传播成功 - 梯度范数: {grad_norm:.4f}")
        else:
            print("❌ 未检测到梯度")
            return False
            
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
        return False
    
    # 测试3: 多次更新
    print("\n📋 测试3: 多次更新测试")
    
    try:
        for i in range(3):
            # 创建新的数据
            new_features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
            new_predictions = torch.randn(batch_size, num_classes, H, W, D, device=device)
            new_labels = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
            
            # 更新
            loss, loss_dict = prototype_manager.update_and_compute_loss(
                features=new_features,
                predictions=new_predictions,
                labels=new_labels,
                is_labeled=True
            )
            
            print(f"   迭代 {i+1}: 总损失 = {loss.item():.4f}")
        
        print("✅ 多次更新测试成功")
        
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
        return False
    
    # 测试4: 无标签数据
    print("\n📋 测试4: 无标签数据测试")
    
    try:
        unlabeled_features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
        unlabeled_predictions = torch.randn(batch_size, num_classes, H, W, D, device=device)
        
        # 模拟高置信度预测
        unlabeled_predictions[:, 1, :, :, :] = 5.0  # 高置信度预测为类别1
        
        loss, loss_dict = prototype_manager.update_and_compute_loss(
            features=unlabeled_features,
            predictions=unlabeled_predictions,
            labels=None,
            is_labeled=False
        )
        
        print(f"✅ 无标签数据测试成功 - 损失: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ 测试4失败: {e}")
        return False
    
    # 测试5: 原型信息检查
    print("\n📋 测试5: 原型信息检查")
    
    try:
        # 获取原型
        prototypes = prototype_manager.get_class_prototypes()
        print(f"✅ 当前原型数量: {len(prototypes)}")
        
        # 获取内存信息
        memory_info = prototype_manager.get_memory_info()
        print(f"✅ 内存状态: 已初始化={memory_info['initialized']}")
        
        if 'inter_class_distances' in memory_info:
            dist_info = memory_info['inter_class_distances']
            print(f"✅ 类间距离统计: 平均={dist_info['mean']:.4f}, 最小={dist_info['min']:.4f}, 最大={dist_info['max']:.4f}")
        
    except Exception as e:
        print(f"❌ 测试5失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！改进的原型管理器工作正常。")
    return True

def compare_memory_usage():
    """比较原始和改进版本的内存使用"""
    print("\n📊 内存使用对比")
    
    from myutils.prototype_manager import PrototypeManager
    
    # 参数
    num_classes = 3
    feature_dim = 64
    elements_per_class = 32
    
    # 原始版本
    original_manager = PrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        k_prototypes=10,
        confidence_threshold=0.8
    )
    
    # 改进版本
    improved_manager = ImprovedPrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        elements_per_class=elements_per_class,
        confidence_threshold=0.8
    )
    
    print(f"原始版本内存: 每类 1 个原型向量 ({feature_dim} 维)")
    print(f"改进版本内存: 每类 {elements_per_class} 个特征向量 ({feature_dim} 维)")
    print(f"内存比例: 改进版本约为原始版本的 {elements_per_class}x")

def main():
    """主函数"""
    print("🚀 改进的原型管理器测试套件")
    print("=" * 50)
    
    # 运行功能测试
    success = test_improved_prototype_manager()
    
    if success:
        # 运行内存对比
        compare_memory_usage()
        
        print("\n💡 使用建议:")
        print("1. 对于资源受限的环境，使用较小的 elements_per_class (8-16)")
        print("2. 对于大数据集，使用标准配置 elements_per_class=32")
        print("3. 调整 confidence_threshold 来平衡特征质量和数量")
        print("4. 监控 inter_class_distances 来评估分离效果")
        
    else:
        print("\n❌ 测试失败，请检查实现")

if __name__ == "__main__":
    main() 