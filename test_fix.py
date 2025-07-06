#!/usr/bin/env python3
"""
简单的维度修复测试
"""

import torch
import torch.nn.functional as F
from myutils.contrastive_prototype_manager import ContrastivePrototypeManager

def test_simple():
    """简单测试"""
    print("🧪 测试维度修复...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建管理器（预期64维，实际16维）
    manager = ContrastivePrototypeManager(
        num_classes=2,
        feature_dim=64,  # 预期维度
        elements_per_class=8,
        use_learned_selector=True,
        device=device
    )
    
    # 创建测试数据（实际16维）
    features = torch.randn(1, 16, 4, 4, 4, device=device)
    predictions = torch.randn(1, 2, 4, 4, 4, device=device)
    labels = torch.randint(0, 2, (1, 4, 4, 4), device=device)
    
    print(f"特征形状: {features.shape}")
    print(f"预期维度: 64, 实际维度: 16")
    
    try:
        # 测试特征提取
        high_quality_features = manager.extract_high_quality_features(
            features, predictions, labels, is_labeled=True
        )
        print("✅ 特征提取成功")
        
        # 测试损失计算
        total_loss, loss_dict = manager.update_and_compute_loss(
            features, predictions, labels, 
            is_labeled=True,
            contrastive_weight=1.0,
            intra_weight=0.1,
            inter_weight=0.1
        )
        print(f"✅ 损失计算成功: {total_loss.item():.4f}")
        
        # 检查选择器
        feature_selectors, memory_selectors = manager.get_selectors()
        if feature_selectors is not None:
            print("✅ 选择器初始化成功")
            for name, selector in feature_selectors.items():
                print(f"  {name}: {selector}")
        
        print("🎉 维度修复测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple() 