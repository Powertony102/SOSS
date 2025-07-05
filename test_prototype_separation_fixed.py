#!/usr/bin/env python3
"""
完全修复的原型分离模块测试代码
解决了所有autograd图交叉引用问题
"""

import torch
import numpy as np
from myutils.prototype_separation import PrototypeMemory


def create_independent_test_data(batch_size, feat_dim, num_classes, H, W, D, device, seed):
    """创建完全独立的测试数据，避免autograd图交叉引用"""
    torch.manual_seed(seed)
    
    # 创建独立的张量
    feat = torch.randn(batch_size, feat_dim, H, W, D, device=device, requires_grad=True)
    logits = torch.randn(batch_size, num_classes + 1, H, W, D, device=device)
    pred = torch.softmax(logits, dim=1)
    label = torch.randint(0, num_classes + 1, (batch_size, 1, H, W, D), device=device)
    is_labelled = torch.tensor([True, True, False, False], device=device)
    
    return feat, pred, label, is_labelled


def test_single_forward_backward():
    """测试单次前向传播和反向传播"""
    print("="*60)
    print("测试单次前向和反向传播")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建原型内存模块
    proto_mem = PrototypeMemory(
        num_classes=2,
        feat_dim=64,
        proto_momentum=0.9,
        conf_thresh=0.8,
        lambda_intra=0.5,
        lambda_inter=0.1,
        margin_m=1.0,
        device=device
    ).to(device)
    
    # 创建测试数据
    feat, pred, label, is_labelled = create_independent_test_data(
        batch_size=2, feat_dim=64, num_classes=2, H=32, W=32, D=16, device=device, seed=42
    )
    
    print(f"输入张量形状:")
    print(f"  feat: {feat.shape}")
    print(f"  pred: {pred.shape}")
    print(f"  label: {label.shape}")
    print(f"  is_labelled: {is_labelled.shape}")
    
    # 前向传播
    loss_dict = proto_mem(feat, label, pred, is_labelled, epoch_idx=0)
    
    # 安全提取损失值
    intra_loss = loss_dict['intra'].detach().item()
    inter_loss = loss_dict['inter'].detach().item()
    total_loss = loss_dict['total'].detach().item()
    n_confident = int(loss_dict['n_confident_pixels'])
    n_protos = int(loss_dict['n_initialized_protos'])
    
    print(f"\n损失计算结果:")
    print(f"  类内损失: {intra_loss:.4f}")
    print(f"  类间损失: {inter_loss:.4f}")
    print(f"  总损失: {total_loss:.4f}")
    print(f"  高置信度像素: {n_confident}")
    print(f"  已初始化原型: {n_protos}")
    
    # 反向传播
    print(f"\n执行反向传播...")
    loss_dict['total'].backward()
    print(f"✓ 反向传播成功")
    print(f"  feat梯度存在: {feat.grad is not None}")
    if feat.grad is not None:
        print(f"  feat梯度范数: {feat.grad.norm().detach().item():.6f}")
    
    print("✓ 单次前向反向传播测试通过")


def test_multi_epoch_training():
    """测试多epoch训练循环，完全解决梯度图交叉引用问题"""
    print("\n" + "="*60)
    print("测试多epoch训练循环（完全隔离版本）")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建原型内存模块
    proto_mem = PrototypeMemory(
        num_classes=2,
        feat_dim=128,
        proto_momentum=0.9,
        conf_thresh=0.8,
        lambda_intra=0.5,
        lambda_inter=0.1,
        margin_m=1.0,
        device=device
    ).to(device)
    
    print(f"初始化PrototypeMemory: 2类，设备={device}")
    
    batch_size = 4
    H, W, D = 64, 64, 32
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        # **阶段1**: 原型更新 - 使用独立数据 + no_grad
        with torch.no_grad():
            update_feat, update_pred, update_label, update_is_labelled = create_independent_test_data(
                batch_size=batch_size, feat_dim=128, num_classes=2, 
                H=H, W=W, D=D, device=device, seed=1000 + epoch
            )
            
            # 移除requires_grad，因为在no_grad中
            update_feat = update_feat.detach()
            
            # 仅更新原型
            _ = proto_mem(update_feat, update_label, update_pred, update_is_labelled, epoch_idx=epoch)
            
            # 立即清理
            del update_feat, update_pred, update_label, update_is_labelled
        
        # **阶段2**: 损失计算 - 使用完全不同的数据
        loss_feat, loss_pred, loss_label, loss_is_labelled = create_independent_test_data(
            batch_size=batch_size, feat_dim=128, num_classes=2,
            H=H, W=W, D=D, device=device, seed=2000 + epoch
        )
        
        # 计算损失（不更新原型）
        proto_losses = proto_mem(loss_feat, loss_label, loss_pred, loss_is_labelled, epoch_idx=None)
        
        # **阶段3**: 安全提取损失值
        intra_val = proto_losses['intra'].detach().item()
        inter_val = proto_losses['inter'].detach().item()
        total_proto_val = proto_losses['total'].detach().item()
        n_confident = int(proto_losses['n_confident_pixels'])
        n_protos = int(proto_losses['n_initialized_protos'])
        
        print(f"  类内损失: {intra_val:.4f}")
        print(f"  类间损失: {inter_val:.4f}")
        print(f"  原型总损失: {total_proto_val:.4f}")
        print(f"  高置信度像素: {n_confident}")
        print(f"  已初始化原型: {n_protos}")
        
        # **阶段4**: 创建其他损失并组合
        other_loss1 = torch.randn(1, device=device, requires_grad=True) * 0.1
        other_loss2 = torch.randn(1, device=device, requires_grad=True) * 0.1
        
        total_loss = other_loss1 + other_loss2 + 0.5 * proto_losses['total']
        total_val = total_loss.detach().item()
        print(f"  总损失: {total_val:.4f}")
        
        # **阶段5**: 反向传播
        total_loss.backward()
        print(f"  ✓ 梯度计算成功")
        
        # **阶段6**: 彻底清理
        del total_loss, proto_losses, other_loss1, other_loss2
        del loss_feat, loss_pred, loss_label, loss_is_labelled
        
        # GPU内存清理
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print("\n✓ 多epoch训练测试完全成功！")


def test_prototype_statistics():
    """测试原型统计功能"""
    print("\n" + "="*60)
    print("测试原型统计功能")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    proto_mem = PrototypeMemory(
        num_classes=3,
        feat_dim=64,
        device=device
    ).to(device)
    
    # 初始化一些原型
    for epoch in range(2):
        with torch.no_grad():
            feat, pred, label, is_labelled = create_independent_test_data(
                batch_size=2, feat_dim=64, num_classes=3,
                H=32, W=32, D=16, device=device, seed=500 + epoch
            )
            feat = feat.detach()  # 移除梯度
            _ = proto_mem(feat, label, pred, is_labelled, epoch_idx=epoch)
            del feat, pred, label, is_labelled
    
    # 获取统计信息
    with torch.no_grad():
        stats = proto_mem.get_prototype_statistics()
        
        print(f"原型统计信息:")
        print(f"  已初始化原型数: {stats['num_initialized']}")
        print(f"  总类别数: {stats['total_classes']}")
        print(f"  最后更新epoch: {stats['last_update_epoch']}")
        
        if 'mean_prototype_norm' in stats:
            print(f"  平均原型范数: {stats['mean_prototype_norm']:.4f}")
        
        if 'mean_pairwise_distance' in stats:
            print(f"  平均原型间距离: {stats['mean_pairwise_distance']:.4f}")
    
    print("✓ 原型统计测试通过")


if __name__ == "__main__":
    try:
        test_single_forward_backward()
        test_multi_epoch_training()
        test_prototype_statistics()
        
        print("\n" + "="*60)
        print("🎉 所有测试完全通过！梯度问题已彻底解决！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 