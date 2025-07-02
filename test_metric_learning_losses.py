#!/usr/bin/env python3
"""
测试度量学习损失函数的实现
验证池内紧凑性损失和池间分离性损失的正确性
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from myutils.cov_dynamic_feature_pool import CovarianceDynamicFeaturePool


def test_metric_learning_losses():
    """测试度量学习损失函数"""
    print("=== 测试度量学习损失函数 ===")
    
    # 设置参数
    feature_dim = 64
    num_dfp = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print(f"特征维度: {feature_dim}")
    print(f"DFP数量: {num_dfp}")
    
    # 创建CovarianceDynamicFeaturePool
    cov_dfp = CovarianceDynamicFeaturePool(
        feature_dim=feature_dim,
        num_dfp=num_dfp,
        max_global_features=1000,
        device=device
    )
    
    # 生成测试数据
    batch_size = 32
    num_global_features = 200
    
    # 1. 生成全局特征并构建DFP
    print("\n1. 生成全局特征并构建DFP...")
    global_features = torch.randn(num_global_features, feature_dim, device=device)
    cov_dfp.add_to_global_pool(global_features)
    
    success = cov_dfp.build_dfps()
    print(f"DFP构建状态: {'成功' if success else '失败'}")
    
    if not success:
        print("DFP构建失败，退出测试")
        return False
    
    # 获取统计信息
    stats = cov_dfp.get_statistics()
    print(f"DFP统计信息: {stats}")
    
    # 2. 测试池内紧凑性损失
    print("\n2. 测试池内紧凑性损失...")
    
    # 创建批次特征，每个DFP分配一些特征
    batch_features_by_dfp = {}
    for dfp_idx in range(num_dfp):
        # 为每个DFP生成一些批次特征
        num_features_in_batch = np.random.randint(3, 10)
        batch_features = torch.randn(num_features_in_batch, feature_dim, device=device)
        batch_features_by_dfp[dfp_idx] = batch_features
    
    # 计算池内紧凑性损失
    loss_compact = cov_dfp.compute_intra_pool_compactness_loss(batch_features_by_dfp)
    print(f"池内紧凑性损失: {loss_compact.item():.6f}")
    print(f"损失是否需要梯度: {loss_compact.requires_grad}")
    
    # 3. 测试池间分离性损失
    print("\n3. 测试池间分离性损失...")
    
    # 使用不同的margin值测试
    margins = [0.5, 1.0, 2.0]
    for margin in margins:
        loss_separate = cov_dfp.compute_inter_pool_separation_loss(margin=margin)
        print(f"池间分离性损失 (margin={margin}): {loss_separate.item():.6f}")
        print(f"损失是否需要梯度: {loss_separate.requires_grad}")
    
    # 4. 测试综合损失函数
    print("\n4. 测试综合损失函数...")
    
    loss_compact, loss_separate = cov_dfp.compute_metric_learning_losses(
        batch_features_by_dfp, margin=1.0
    )
    print(f"综合 - 池内紧凑性损失: {loss_compact.item():.6f}")
    print(f"综合 - 池间分离性损失: {loss_separate.item():.6f}")
    
    # 5. 测试特征分组功能
    print("\n5. 测试特征分组功能...")
    
    # 生成测试特征和预测
    test_features = torch.randn(batch_size, feature_dim, device=device)
    test_predictions = torch.randint(0, num_dfp, (batch_size,), device=device)
    
    grouped_features = cov_dfp.group_features_by_dfp_predictions(test_features, test_predictions)
    
    print(f"分组结果:")
    for dfp_idx, features in grouped_features.items():
        if features is not None:
            print(f"  DFP {dfp_idx}: {features.shape[0]} 个特征")
        else:
            print(f"  DFP {dfp_idx}: 无特征")
    
    # 6. 测试梯度计算
    print("\n6. 测试梯度计算...")
    
    # 创建需要梯度的特征
    test_features_grad = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
    test_predictions_grad = torch.randint(0, num_dfp, (batch_size,), device=device)
    
    grouped_features_grad = cov_dfp.group_features_by_dfp_predictions(test_features_grad, test_predictions_grad)
    
    loss_compact_grad, loss_separate_grad = cov_dfp.compute_metric_learning_losses(
        grouped_features_grad, margin=1.0
    )
    
    total_loss = loss_compact_grad + loss_separate_grad
    
    print(f"总损失: {total_loss.item():.6f}")
    print(f"总损失是否需要梯度: {total_loss.requires_grad}")
    
    # 反向传播测试
    total_loss.backward()
    
    if test_features_grad.grad is not None:
        print(f"梯度计算成功，梯度范数: {test_features_grad.grad.norm().item():.6f}")
    else:
        print("梯度计算失败")
    
    # 7. 测试边界情况
    print("\n7. 测试边界情况...")
    
    # 空的batch_features_by_dfp
    empty_features = {}
    loss_compact_empty = cov_dfp.compute_intra_pool_compactness_loss(empty_features)
    print(f"空批次的池内紧凑性损失: {loss_compact_empty.item():.6f}")
    
    # 单个特征的情况
    single_feature = {0: torch.randn(1, feature_dim, device=device)}
    loss_compact_single = cov_dfp.compute_intra_pool_compactness_loss(single_feature)
    print(f"单个特征的池内紧凑性损失: {loss_compact_single.item():.6f}")
    
    print("\n=== 测试完成 ===")
    return True


def test_dfp_centers_distance():
    """测试DFP中心之间的距离"""
    print("\n=== 测试DFP中心距离 ===")
    
    feature_dim = 32
    num_dfp = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cov_dfp = CovarianceDynamicFeaturePool(
        feature_dim=feature_dim,
        num_dfp=num_dfp,
        max_global_features=1000,
        device=device
    )
    
    # 生成明显分离的特征集合
    cluster_centers = []
    for i in range(num_dfp):
        center = torch.zeros(feature_dim, device=device)
        center[i] = 5.0  # 在不同维度上设置不同的中心
        cluster_centers.append(center)
    
    # 为每个簇生成特征
    all_features = []
    for i, center in enumerate(cluster_centers):
        cluster_features = center.unsqueeze(0) + 0.1 * torch.randn(50, feature_dim, device=device)
        all_features.append(cluster_features)
    
    global_features = torch.cat(all_features, dim=0)
    cov_dfp.add_to_global_pool(global_features)
    
    success = cov_dfp.build_dfps()
    print(f"DFP构建状态: {'成功' if success else '失败'}")
    
    if success:
        # 检查DFP之间的距离
        centers = cov_dfp.dfp_centers
        print(f"DFP中心形状: {centers.shape}")
        
        for i in range(num_dfp):
            for j in range(i + 1, num_dfp):
                distance = torch.norm(centers[i] - centers[j])
                print(f"DFP {i} 与 DFP {j} 之间的距离: {distance.item():.6f}")
        
        # 测试不同margin值的分离损失
        for margin in [0.1, 1.0, 5.0, 10.0]:
            loss_separate = cov_dfp.compute_inter_pool_separation_loss(margin=margin)
            print(f"分离损失 (margin={margin}): {loss_separate.item():.6f}")


if __name__ == "__main__":
    print("开始测试度量学习损失函数实现...")
    
    try:
        success = test_metric_learning_losses()
        if success:
            test_dfp_centers_distance()
            print("\n✅ 所有测试通过！")
        else:
            print("\n❌ 测试失败！")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc() 