#!/usr/bin/env python3
"""
测试动态特征维度推断功能
验证PrototypeMemory能够正确处理不同的输入特征维度
"""

import torch
import torch.nn.functional as F
import logging
from myutils.prototype_separation import PrototypeMemory

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_dynamic_feat_dim():
    """测试动态特征维度推断"""
    print("=" * 60)
    print("测试动态特征维度推断功能")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试不同的特征维度
    test_dims = [16, 32, 64, 128]
    
    for feat_dim in test_dims:
        print(f"\n--- 测试特征维度: {feat_dim} ---")
        
        # 创建PrototypeMemory，不指定feat_dim
        proto_mem = PrototypeMemory(
            num_classes=2,  # 2个前景类
            feat_dim=None,  # 关键：设为None，运行时推断
            proto_momentum=0.9,
            conf_thresh=0.8,
            lambda_intra=1.0,
            lambda_inter=0.1,
            margin_m=1.0,
            device=device
        ).to(device)
        
        # 创建测试数据
        batch_size = 2
        H, W, D = 8, 8, 8
        num_classes = 3  # 包括背景
        
        # 特征张量 - 使用当前测试的维度
        feat = torch.randn(batch_size, feat_dim, H, W, D, device=device, requires_grad=True)
        
        # 预测张量
        pred_logits = torch.randn(batch_size, num_classes, H, W, D, device=device, requires_grad=True)
        pred = F.softmax(pred_logits, dim=1)
        
        # 标签张量
        label = torch.randint(0, num_classes, (batch_size, 1, H, W, D), device=device)
        
        # is_labelled掩码
        is_labelled = torch.tensor([True, False], device=device)
        
        print(f"输入特征形状: {feat.shape}")
        print(f"预测形状: {pred.shape}")
        print(f"标签形状: {label.shape}")
        
        try:
            # 第一次forward - 应该自动推断特征维度
            losses = proto_mem(
                feat=feat,
                label=label,
                pred=pred,
                is_labelled=is_labelled,
                epoch_idx=0
            )
            
            print(f"✓ 成功推断特征维度: {proto_mem.feat_dim}")
            print(f"✓ 原型形状: {proto_mem.prototypes.shape}")
            print(f"✓ 损失计算成功:")
            print(f"  - loss_intra: {losses['intra'].item():.4f}")
            print(f"  - loss_inter: {losses['inter'].item():.4f}")
            print(f"  - loss_total: {losses['total'].item():.4f}")
            print(f"  - n_confident_pixels: {losses['n_confident_pixels']}")
            
            # 测试梯度反向传播
            total_loss = losses['total']
            total_loss.backward()
            print(f"✓ 梯度反向传播成功")
            
            # 检查梯度
            if feat.grad is not None:
                print(f"✓ 特征梯度非零: {feat.grad.abs().sum().item():.6f}")
            
            # 第二次forward - 应该使用已推断的维度
            with torch.no_grad():
                feat2 = torch.randn(batch_size, feat_dim, H, W, D, device=device)
                pred2 = F.softmax(torch.randn(batch_size, num_classes, H, W, D, device=device), dim=1)
                
                losses2 = proto_mem(
                    feat=feat2,
                    label=label,
                    pred=pred2,
                    is_labelled=is_labelled,
                    epoch_idx=1
                )
                print(f"✓ 第二次forward成功: {losses2['total'].item():.4f}")
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("动态特征维度推断测试完成")
    print("=" * 60)

def test_dimension_mismatch_error():
    """测试维度不匹配时的错误处理"""
    print("\n" + "=" * 60)
    print("测试维度不匹配错误处理")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建PrototypeMemory，指定特定的feat_dim
    proto_mem = PrototypeMemory(
        num_classes=2,
        feat_dim=64,  # 指定为64
        proto_momentum=0.9,
        conf_thresh=0.8,
        device=device
    ).to(device)
    
    # 创建不匹配维度的输入
    batch_size = 2
    H, W, D = 8, 8, 8
    wrong_feat_dim = 32  # 与指定的64不匹配
    
    feat = torch.randn(batch_size, wrong_feat_dim, H, W, D, device=device)
    pred = F.softmax(torch.randn(batch_size, 3, H, W, D, device=device), dim=1)
    label = torch.randint(0, 3, (batch_size, 1, H, W, D), device=device)
    is_labelled = torch.tensor([True, False], device=device)
    
    print(f"原型期望维度: {proto_mem.feat_dim}")
    print(f"输入特征维度: {wrong_feat_dim}")
    
    try:
        losses = proto_mem(
            feat=feat,
            label=label,
            pred=pred,
            is_labelled=is_labelled,
            epoch_idx=0
        )
        print("✗ 预期的维度不匹配错误没有发生")
    except RuntimeError as e:
        if "特征维度不匹配" in str(e):
            print(f"✓ 正确捕获维度不匹配错误: {e}")
        else:
            print(f"✗ 意外的RuntimeError: {e}")
    except Exception as e:
        print(f"✗ 意外的错误类型: {e}")

def test_integration_simulation():
    """模拟实际训练集成场景"""
    print("\n" + "=" * 60)
    print("模拟实际训练集成场景")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模拟LA数据集配置
    proto_mem = PrototypeMemory(
        num_classes=1,  # LA数据集：1个前景类
        feat_dim=None,  # 运行时推断
        proto_momentum=0.95,
        conf_thresh=0.85,
        lambda_intra=0.3,
        lambda_inter=0.1,
        margin_m=1.5,
        device=device
    ).to(device)
    
    # 模拟VNet输出特征（通常是16维）
    actual_feat_dim = 16  # VNet decoder输出维度
    batch_size = 4
    H, W, D = 16, 16, 16
    
    print(f"模拟VNet特征维度: {actual_feat_dim}")
    
    # 模拟多个训练步骤
    for step in range(5):
        print(f"\n--- 训练步骤 {step + 1} ---")
        
        # 创建模拟数据
        feat = torch.randn(batch_size, actual_feat_dim, H, W, D, device=device, requires_grad=True)
        pred_logits = torch.randn(batch_size, 2, H, W, D, device=device, requires_grad=True)  # 2类：背景+前景
        pred = F.softmax(pred_logits, dim=1)
        label = torch.randint(0, 2, (batch_size, 1, H, W, D), device=device)
        is_labelled = torch.tensor([True, True, False, False], device=device)  # 前2个有标签
        
        try:
            # 损失计算阶段（不更新原型）
            losses = proto_mem(
                feat=feat,
                label=label,
                pred=pred,
                is_labelled=is_labelled,
                epoch_idx=None  # 不更新原型
            )
            
            # 模拟总损失和反向传播
            supervised_loss = torch.randn(1, device=device, requires_grad=True)
            consistency_loss = torch.randn(1, device=device, requires_grad=True)
            
            total_loss = supervised_loss + consistency_loss + 0.3 * losses['total']
            total_loss.backward()
            
            print(f"✓ 损失计算成功: {losses['total'].item():.4f}")
            print(f"  - 置信像素数: {losses['n_confident_pixels']}")
            print(f"  - 已初始化原型数: {losses['n_initialized_protos']}")
            
            # 模拟optimizer.step()后的原型更新
            if step % 2 == 0:  # 每2步更新一次原型
                with torch.no_grad():
                    update_feat = feat.detach().clone()
                    update_pred = pred.detach().clone()
                    
                    _ = proto_mem(
                        feat=update_feat,
                        label=label,
                        pred=update_pred,
                        is_labelled=is_labelled,
                        epoch_idx=step
                    )
                    print(f"✓ 原型更新完成")
                    
                    # 获取统计信息
                    stats = proto_mem.get_prototype_statistics()
                    print(f"  - 原型统计: {stats['num_initialized']}/{stats['total_classes']} 已初始化")
            
        except Exception as e:
            print(f"✗ 步骤 {step + 1} 失败: {e}")
            break
    
    print(f"\n✓ 集成模拟测试完成")
    print(f"最终原型维度: {proto_mem.feat_dim}")

if __name__ == "__main__":
    # 运行所有测试
    test_dynamic_feat_dim()
    test_dimension_mismatch_error()
    test_integration_simulation()
    
    print("\n🎉 所有测试完成！") 