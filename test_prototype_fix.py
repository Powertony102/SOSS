#!/usr/bin/env python3
"""
测试脚本，演示原型损失为零问题的修复
"""

import torch
import logging
from myutils.prototype_separation_fixed import PrototypeMemoryFixed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_prototype_fix():
    """测试修复的原型内存模块"""
    print("="*60)
    print("测试修复版本的 PrototypeMemory 模块")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建带有自适应阈值的修复原型内存
    proto_mem = PrototypeMemoryFixed(
        num_classes=1,  # LA数据集：1个前景类
        feat_dim=64,
        proto_momentum=0.9,
        conf_thresh=0.3,  # 从较低阈值开始
        conf_thresh_max=0.85,  # 逐渐增加到这个值
        conf_thresh_rampup=1000,  # 在1000次迭代内
        lambda_intra=0.3,
        lambda_inter=0.1,
        margin_m=1.5,
        min_pixels_per_class=5,  # 确保每类至少5个像素
        use_labeled_fallback=True,  # 使用标注像素作为后备
        device=device
    ).to(device)
    
    print(f"创建了带有自适应阈值的 PrototypeMemoryFixed")
    
    # 模拟训练早期的糟糕预测
    batch_size = 2
    H, W, D = 32, 32, 16
    
    for iteration in range(0, 1100, 100):
        print(f"\n--- 迭代 {iteration} ---")
        
        # 创建模拟糟糕早期预测的测试数据
        feat = torch.randn(batch_size, 64, H, W, D, device=device, requires_grad=True)
        
        # 模拟糟糕的预测（低置信度）
        if iteration < 500:
            # 早期训练：非常糟糕的预测
            logits = torch.randn(batch_size, 2, H, W, D, device=device) * 0.5
        else:
            # 后期训练：更好的预测
            logits = torch.randn(batch_size, 2, H, W, D, device=device) * 2.0
        
        pred = torch.softmax(logits, dim=1)
        label = torch.randint(0, 2, (batch_size, 1, H, W, D), device=device)
        is_labelled = torch.tensor([True, False], device=device)
        
        # 计算损失
        losses = proto_mem(
            feat=feat,
            label=label,
            pred=pred,
            is_labelled=is_labelled,
            epoch_idx=iteration
        )
        
        print(f"  置信度阈值: {losses['current_conf_thresh']:.3f}")
        print(f"  最大置信度: {losses['max_confidence']:.3f}")
        print(f"  平均置信度: {losses['mean_confidence']:.3f}")
        print(f"  置信像素: {losses['n_confident_pixels']}")
        print(f"  已初始化原型: {losses['n_initialized_protos']}")
        print(f"  类内损失: {losses['intra'].item():.6f}")
        print(f"  类间损失: {losses['inter'].item():.6f}")
        print(f"  总损失: {losses['total'].item():.6f}")
        
        # 测试反向传播
        if losses['total'].item() > 0:
            losses['total'].backward()
            print(f"  ✓ 反向传播成功")
            if feat.grad is not None:
                print(f"  ✓ 梯度已计算: {feat.grad.norm().item()