#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证池间分离性损失修复的测试脚本
"""

import torch
import numpy as np
from myutils.cov_dynamic_feature_pool import CovarianceDynamicFeaturePool

def test_separation_loss_dynamics():
    """测试分离损失是否能够动态变化"""
    
    print("=== 测试池间分离性损失动态变化 ===")
    
    # 设置参数
    feature_dim = 64
    num_dfp = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建CovarianceDynamicFeaturePool
    cov_dfp = CovarianceDynamicFeaturePool(
        feature_dim=feature_dim,
        num_dfp=num_dfp,
        max_global_features=1000,
        device=device
    )
    
    # 1. 添加全局特征
    print("1. 添加全局特征...")
    global_features = torch.randn(200, feature_dim, device=device)
    cov_dfp.add_to_global_pool(global_features)
    
    # 2. 构建DFPs
    print("2. 构建DFPs...")
    success = cov_dfp.build_dfps()
    if not success:
        print("DFP构建失败!")
        return
    
    # 3. 计算初始分离损失
    print("3. 计算初始分离损失...")
    initial_loss = cov_dfp.compute_inter_pool_separation_loss(margin=1.0)
    print(f"初始分离损失: {initial_loss.item():.6f}")
    
    # 4. 模拟多次训练迭代，观察损失变化
    print("4. 模拟训练迭代...")
    losses = []
    
    for iter_num in range(10):
        # 生成批次特征
        batch_features = torch.randn(32, feature_dim, device=device, requires_grad=True)
        
        # 模拟Selector预测（随机分配到不同DFP）
        dfp_predictions = torch.randint(0, num_dfp, (32,), device=device)
        
        # 按DFP分组特征
        batch_features_by_dfp = cov_dfp.group_features_by_dfp_predictions(
            batch_features, dfp_predictions
        )
        
        # 计算分离损失（此时损失应该能参与梯度更新）
        separation_loss = cov_dfp.compute_inter_pool_separation_loss(margin=1.0)
        
        # 模拟梯度更新（实际训练中，这个损失会作为总损失的一部分）
        if batch_features.requires_grad:
            fake_total_loss = separation_loss + torch.sum(batch_features ** 2) * 0.001
            fake_total_loss.backward()
        
        # 更新DFPs
        with torch.no_grad():
            cov_dfp.update_dfps_with_batch_features(batch_features_by_dfp, 
                                                   update_rate=0.1, 
                                                   max_dfp_size=100)
            
        # 重新计算分离损失
        new_separation_loss = cov_dfp.compute_inter_pool_separation_loss(margin=1.0)
        losses.append(new_separation_loss.item())
        
        print(f"迭代 {iter_num+1}: 分离损失 = {new_separation_loss.item():.6f}")
    
    # 5. 分析结果
    print(f"\n5. 结果分析:")
    print(f"初始损失: {initial_loss.item():.6f}")
    print(f"最终损失: {losses[-1]:.6f}")
    print(f"损失变化: {losses[-1] - initial_loss.item():.6f}")
    
    # 检查损失是否有变化
    loss_std = np.std(losses)
    print(f"损失标准差: {loss_std:.6f}")
    
    if loss_std > 1e-6:
        print("✅ 修复成功! 分离损失能够动态变化")
    else:
        print("❌ 修复失败! 分离损失仍然固定不变")
    
    # 6. 详细的中心距离分析
    print(f"\n6. DFP中心距离分析:")
    
    # 计算当前所有DFP中心
    current_centers = []
    for i in range(num_dfp):
        if cov_dfp.dfps[i] is not None and cov_dfp.dfps[i].shape[0] > 0:
            center = torch.mean(cov_dfp.dfps[i], dim=0)
            current_centers.append(center)
        else:
            center = torch.zeros(feature_dim, device=device)
            current_centers.append(center)
    
    # 计算所有中心对之间的距离
    distances = []
    for i in range(num_dfp):
        for j in range(i + 1, num_dfp):
            distance = torch.norm(current_centers[i] - current_centers[j]).item()
            distances.append(distance)
            print(f"DFP {i} - DFP {j} 中心距离: {distance:.4f}")
    
    print(f"平均中心距离: {np.mean(distances):.4f}")
    print(f"最小中心距离: {np.min(distances):.4f}")
    print(f"最大中心距离: {np.max(distances):.4f}")

def test_gradient_flow():
    """测试梯度是否能正确流动"""
    
    print("\n=== 测试梯度流动 ===")
    
    feature_dim = 32
    num_dfp = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建CovarianceDynamicFeaturePool
    cov_dfp = CovarianceDynamicFeaturePool(
        feature_dim=feature_dim,
        num_dfp=num_dfp,
        max_global_features=500,
        device=device
    )
    
    # 初始化DFPs
    global_features = torch.randn(100, feature_dim, device=device)
    cov_dfp.add_to_global_pool(global_features)
    cov_dfp.build_dfps()
    
    # 创建需要梯度的特征
    batch_features = torch.randn(16, feature_dim, device=device, requires_grad=True)
    dfp_predictions = torch.randint(0, num_dfp, (16,), device=device)
    
    # 分组特征
    batch_features_by_dfp = cov_dfp.group_features_by_dfp_predictions(
        batch_features, dfp_predictions
    )
    
    # 计算分离损失
    separation_loss = cov_dfp.compute_inter_pool_separation_loss(margin=2.0)
    
    print(f"分离损失: {separation_loss.item():.6f}")
    print(f"损失是否需要梯度: {separation_loss.requires_grad}")
    
    # 尝试反向传播
    try:
        separation_loss.backward()
        if batch_features.grad is not None:
            grad_norm = torch.norm(batch_features.grad).item()
            print(f"✅ 梯度计算成功! 梯度范数: {grad_norm:.6f}")
        else:
            print("❌ 没有计算出梯度")
    except Exception as e:
        print(f"❌ 梯度计算失败: {e}")

if __name__ == "__main__":
    print("开始验证池间分离性损失修复...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    test_separation_loss_dynamics()
    test_gradient_flow()
    
    print("\n测试完成!") 