#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from myutils.prototype_manager import PrototypeManager

def test_prototype_manager():
    """测试原型管理器的基本功能"""
    print("测试原型管理器...")
    
    # 设置参数
    num_classes = 2
    feature_dim = 64
    batch_size = 2
    H, W, D = 16, 16, 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 创建原型管理器
    prototype_manager = PrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        k_prototypes=5,
        confidence_threshold=0.8,
        device=device
    )
    
    # 创建模拟数据
    features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
    outputs = torch.randn(batch_size, num_classes, H, W, D, device=device, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
    
    print(f"特征形状: {features.shape}")
    print(f"输出形状: {outputs.shape}")
    print(f"标签形状: {labels.shape}")
    
    try:
        # 测试原型分离损失计算
        print("\n测试标记数据的原型分离损失...")
        loss_labeled, loss_dict_labeled = prototype_manager.update_and_compute_loss(
            features, outputs, labels, 
            is_labeled=True,
            intra_weight=1.0,
            inter_weight=0.1,
            margin=1.0
        )
        
        print(f"标记数据损失: {loss_labeled.item():.6f}")
        print(f"损失详情: {loss_dict_labeled}")
        
        # 测试反向传播
        print("\n测试反向传播...")
        loss_labeled.backward()
        print("标记数据反向传播成功！")
        
        # 检查梯度
        if features.grad is not None:
            print(f"特征梯度范数: {torch.norm(features.grad).item():.6f}")
        else:
            print("警告：特征没有梯度！")
        
        # 重置梯度
        features.grad = None
        outputs.grad = None
        
        # 测试未标记数据
        print("\n测试未标记数据的原型分离损失...")
        loss_unlabeled, loss_dict_unlabeled = prototype_manager.compute_prototype_loss(
            features, outputs,
            is_labeled=False,
            intra_weight=1.0,
            inter_weight=0.1,
            margin=1.0
        )
        
        print(f"未标记数据损失: {loss_unlabeled.item():.6f}")
        print(f"损失详情: {loss_dict_unlabeled}")
        
        # 测试反向传播
        print("\n测试未标记数据反向传播...")
        loss_unlabeled.backward()
        print("未标记数据反向传播成功！")
        
        # 检查梯度
        if features.grad is not None:
            print(f"特征梯度范数: {torch.norm(features.grad).item():.6f}")
        else:
            print("警告：特征没有梯度！")
        
        # 测试原型信息
        print("\n原型信息:")
        prototype_info = prototype_manager.get_prototype_info()
        for key, value in prototype_info.items():
            print(f"  {key}: {value}")
        
        print("\n✅ 所有测试通过！原型分离功能正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_combined_loss():
    """测试组合损失的反向传播"""
    print("\n" + "="*50)
    print("测试组合损失反向传播...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 2
    feature_dim = 64
    batch_size = 2
    H, W, D = 16, 16, 16
    
    # 创建原型管理器
    prototype_manager = PrototypeManager(
        num_classes=num_classes,
        feature_dim=feature_dim,
        device=device
    )
    
    # 创建模拟数据
    features = torch.randn(batch_size, feature_dim, H, W, D, device=device, requires_grad=True)
    outputs = torch.randn(batch_size, num_classes, H, W, D, device=device, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size, H, W, D), device=device)
    
    try:
        # 模拟多个损失项
        print("计算监督损失...")
        pred_probs = F.softmax(outputs, dim=1)
        supervised_loss = F.cross_entropy(outputs.view(-1, num_classes), labels.view(-1))
        
        print("计算一致性损失...")
        consistency_loss = F.mse_loss(pred_probs, pred_probs)  # 简化的一致性损失
        
        print("计算原型分离损失...")
        prototype_loss, _ = prototype_manager.update_and_compute_loss(
            features, outputs, labels,
            is_labeled=True
        )
        
        # 组合损失
        total_loss = 0.5 * supervised_loss + 1.0 * consistency_loss + 1.0 * prototype_loss
        
        print(f"监督损失: {supervised_loss.item():.6f}")
        print(f"一致性损失: {consistency_loss.item():.6f}")
        print(f"原型损失: {prototype_loss.item():.6f}")
        print(f"总损失: {total_loss.item():.6f}")
        
        # 测试反向传播
        print("\n测试组合损失反向传播...")
        total_loss.backward()
        print("✅ 组合损失反向传播成功！")
        
        # 检查梯度
        if features.grad is not None:
            print(f"特征梯度范数: {torch.norm(features.grad).item():.6f}")
        if outputs.grad is not None:
            print(f"输出梯度范数: {torch.norm(outputs.grad).item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 组合损失测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始测试原型分离功能修复...")
    
    # 基本功能测试
    success1 = test_prototype_manager()
    
    # 组合损失测试
    success2 = test_combined_loss()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！原型分离功能已正确修复。")
        print("现在可以安全地运行训练脚本了。")
    else:
        print("\n💥 测试失败！请检查修复。")
        exit(1) 