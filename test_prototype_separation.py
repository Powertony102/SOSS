#!/usr/bin/env python3
"""
Unit tests and example usage for PrototypeMemory module.
"""

import torch
import numpy as np
from myutils.prototype_separation import PrototypeMemory


def test_prototype_memory():
    """Unit test for PrototypeMemory module."""
    print("Running PrototypeMemory unit tests...")
    
    # Test parameters
    batch_size = 2
    feat_dim = 64
    num_classes = 2  # Foreground classes (excluding background)
    H, W, D = 112, 112, 80
    K = num_classes + 1  # Include background class
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create module
    proto_mem = PrototypeMemory(
        num_classes=num_classes,
        feat_dim=feat_dim,
        proto_momentum=0.9,
        conf_thresh=0.7,
        lambda_intra=1.0,
        lambda_inter=0.1,
        margin_m=1.0,
        device=device
    ).to(device)
    
    # Create test data
    torch.manual_seed(42)
    feat = torch.randn(batch_size, feat_dim, H, W, D, device=device, requires_grad=True)
    
    # Create realistic predictions (softmax)
    logits = torch.randn(batch_size, K, H, W, D, device=device)
    pred = torch.softmax(logits, dim=1)
    
    # Create labels (with some background pixels)
    label = torch.randint(0, K, (batch_size, 1, H, W, D), device=device)
    
    # Mark first sample as labelled, second as unlabelled
    is_labelled = torch.tensor([True, False], device=device)
    
    print(f"Input shapes:")
    print(f"  feat: {feat.shape}")
    print(f"  pred: {pred.shape}")
    print(f"  label: {label.shape}")
    print(f"  is_labelled: {is_labelled.shape}")
    
    # Test forward pass
    loss_dict = proto_mem(feat, label, pred, is_labelled, epoch_idx=0)
    
    # Detach all values for printing BEFORE backward
    print(f"\nLoss computation:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.detach().item():.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Test backward pass
    total_loss = loss_dict['total']
    print(f"\nBackward pass test:")
    print(f"  total_loss.requires_grad: {total_loss.requires_grad}")
    
    if total_loss.requires_grad:
        total_loss.backward()
        print(f"  feat.grad is not None: {feat.grad is not None}")
        if feat.grad is not None:
            print(f"  feat.grad.norm(): {feat.grad.norm().detach().item():.6f}")
    
    # Test prototype statistics
    stats = proto_mem.get_prototype_statistics()
    print(f"\nPrototype statistics:")
    for key, value in stats.items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 5:
            print(f"  {key}: array of length {len(value)}")
        else:
            print(f"  {key}: {value}")
    
    # Test multiple epochs
    print(f"\nMulti-epoch test:")
    for epoch in range(1, 4):
        # 为每个epoch创建完全独立的测试数据
        torch.manual_seed(100 + epoch)  # 每个epoch不同的随机种子
        with torch.no_grad():
            # 创建全新的独立张量
            feat_new = torch.randn(batch_size, feat_dim, H, W, D, device=device)
            logits_new = torch.randn(batch_size, K, H, W, D, device=device)
            pred_new = torch.softmax(logits_new, dim=1)
            label_new = torch.randint(0, K, (batch_size, 1, H, W, D), device=device)
            is_labelled_new = torch.tensor([True, False], device=device)
            
            loss_dict_new = proto_mem(feat_new, label_new, pred_new, is_labelled_new, epoch_idx=epoch)
            total_loss_val = loss_dict_new['total'].detach().item()
            n_confident = int(loss_dict_new['n_confident_pixels'])
            n_protos = int(loss_dict_new['n_initialized_protos'])
            
            print(f"  Epoch {epoch} - total_loss: {total_loss_val:.6f}, "
                  f"n_confident: {n_confident}, "
                  f"n_protos: {n_protos}")
            
            # 清理张量
            del feat_new, logits_new, pred_new, label_new, is_labelled_new, loss_dict_new
    
    print("✓ All tests passed!")


def example_usage():
    """Example usage in training loop."""
    print("\n" + "="*60)
    print("Example Usage in Training Loop")
    print("="*60)

    num_classes = 2
    feat_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    proto_mem = PrototypeMemory(
        num_classes=num_classes,
        feat_dim=feat_dim,
        proto_momentum=0.9,
        conf_thresh=0.8,
        lambda_intra=0.5,
        lambda_inter=0.1,
        margin_m=1.0,
        device=device
    ).to(device)

    print(f"Initialized PrototypeMemory with {num_classes} classes on {device}")

    batch_size = 4
    H, W, D = 64, 64, 32

    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")

        # **STEP 1**: 原型更新阶段 - 使用完全独立的张量和no_grad
        torch.manual_seed(42 + epoch * 100)  # 每个epoch不同的随机种子
        with torch.no_grad():
            # 创建第一套完全独立的张量（仅用于原型更新）
            update_features = torch.randn(batch_size, feat_dim, H, W, D, device=device)
            update_logits = torch.randn(batch_size, num_classes + 1, H, W, D, device=device)
            update_predictions = torch.softmax(update_logits, dim=1)
            update_labels = torch.randint(0, num_classes + 1, (batch_size, 1, H, W, D), device=device)
            update_is_labelled = torch.tensor([True, True, False, False], device=device)
            
            # 仅更新原型，不计算梯度
            _ = proto_mem(
                feat=update_features,
                label=update_labels,
                pred=update_predictions,
                is_labelled=update_is_labelled,
                epoch_idx=epoch
            )

        # **STEP 2**: 损失计算阶段 - 使用第二套完全独立的张量
        torch.manual_seed(100 + epoch * 200)  # 另一个随机种子确保完全不同的数据
        
        # 创建第二套完全独立的张量（用于梯度计算）
        loss_features = torch.randn(batch_size, feat_dim, H, W, D, device=device, requires_grad=True)
        loss_logits = torch.randn(batch_size, num_classes + 1, H, W, D, device=device)
        loss_predictions = torch.softmax(loss_logits, dim=1)
        loss_labels = torch.randint(0, num_classes + 1, (batch_size, 1, H, W, D), device=device)
        loss_is_labelled = torch.tensor([True, True, False, False], device=device)

        # 计算损失（不更新原型）
        proto_losses = proto_mem(
            feat=loss_features,
            label=loss_labels,
            pred=loss_predictions,
            is_labelled=loss_is_labelled,
            epoch_idx=None  # 关键：不更新原型
        )

        # **STEP 3**: 安全地提取损失值（在backward之前detach）
        intra_loss_val = proto_losses['intra'].detach().item()
        inter_loss_val = proto_losses['inter'].detach().item()
        total_proto_loss_val = proto_losses['total'].detach().item()
        n_confident = int(proto_losses['n_confident_pixels'])
        n_protos = int(proto_losses['n_initialized_protos'])
        
        print(f"  Intra-class loss: {intra_loss_val:.4f}")
        print(f"  Inter-class loss: {inter_loss_val:.4f}")
        print(f"  Total proto loss: {total_proto_loss_val:.4f}")
        print(f"  Confident pixels: {n_confident}")
        print(f"  Initialized prototypes: {n_protos}")

        # **STEP 4**: 创建完全独立的其他损失项
        dice_loss = torch.randn(1, device=device, requires_grad=True) * 0.1
        consistency_loss = torch.randn(1, device=device, requires_grad=True) * 0.1
        
        # 组合总损失
        total_loss = dice_loss + 0.1 * consistency_loss + 0.5 * proto_losses['total']
        total_loss_val = total_loss.detach().item()
        print(f"  Combined total loss: {total_loss_val:.4f}")

        # **STEP 5**: 执行backward
        total_loss.backward()
        print(f"  Gradients computed successfully")

        # **STEP 6**: 清理计算图（可选，但有助于内存管理）
        del total_loss, proto_losses, dice_loss, consistency_loss
        del loss_features, loss_logits, loss_predictions, loss_labels, loss_is_labelled
        del update_features, update_logits, update_predictions, update_labels, update_is_labelled
        
        # 显式触发垃圾回收
        if device == 'cuda':
            torch.cuda.empty_cache()

        # **STEP 7**: 原型统计（使用no_grad确保安全）
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                stats = proto_mem.get_prototype_statistics()
                num_init = int(stats['num_initialized'])
                total_cls = int(stats['total_classes'])
                print(f"  Prototype stats: {num_init}/{total_cls} initialized")
                if 'mean_pairwise_distance' in stats:
                    mean_dist = float(stats['mean_pairwise_distance'])
                    print(f"  Mean prototype distance: {mean_dist:.4f}")
    
    print("\n✓ Multi-epoch training completed successfully!")


def integration_example():
    """Example of integrating PrototypeMemory into existing training framework."""
    print("\n" + "="*60)
    print("Integration Example with Existing Framework")
    print("="*60)
    
    # 模拟您当前的训练框架中的集成
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化原型内存模块
    proto_memory = PrototypeMemory(
        num_classes=2,  # LA数据集的前景类别数 (不含背景)
        feat_dim=64,    # 与您的embedding_dim保持一致
        proto_momentum=0.95,    # 较高的动量用于稳定更新
        conf_thresh=0.85,       # 高置信度阈值
        lambda_intra=0.3,       # 类内紧致性权重
        lambda_inter=0.1,       # 类间分离权重
        margin_m=1.5,           # 分离边界
        device=device
    ).to(device)
    
    print("PrototypeMemory配置:")
    print(f"  num_classes: {proto_memory.num_classes}")
    print(f"  feat_dim: {proto_memory.feat_dim}")
    print(f"  proto_momentum: {proto_memory.proto_momentum}")
    print(f"  conf_thresh: {proto_memory.conf_thresh}")
    print(f"  lambda_intra: {proto_memory.lambda_intra}")
    print(f"  lambda_inter: {proto_memory.lambda_inter}")
    print(f"  margin_m: {proto_memory.margin_m}")
    
    # 在训练循环中的使用示例
    print(f"\n在训练循环中集成PrototypeMemory:")
    print("```python")
    print("# 在train_cov_dfp_3d.py中的集成示例")
    print("def train_with_prototype_separation(model, sampled_batch, optimizer, ...):")
    print("    # ... 现有的前向传播代码 ...")
    print("    outputs_v, outputs_a, embedding_v, embedding_a = model(volume_batch)")
    print("    ")
    print("    # 获取解码器特征 (假设是embedding_v)")
    print("    decoder_features = embedding_v  # (B, C, H, W, D)")
    print("    predictions = torch.softmax(outputs_v, dim=1)  # (B, K, H, W, D)")
    print("    ")
    print("    # 计算原型分离损失")
    print("    proto_losses = proto_memory(")
    print("        feat=decoder_features,")
    print("        label=label_batch,")
    print("        pred=predictions,")
    print("        is_labelled=is_labelled_mask,")
    print("        epoch_idx=current_epoch")
    print("    )")
    print("    ")
    print("    # 合并到总损失中")
    print("    total_loss = (args.lamda * loss_s + ")
    print("                  lambda_c * loss_c + ")
    print("                  args.lambda_hcc * loss_hcc +")
    print("                  0.5 * proto_losses['total'])  # 添加原型损失")
    print("```")


if __name__ == "__main__":
    test_prototype_memory()
    example_usage()
    integration_example() 