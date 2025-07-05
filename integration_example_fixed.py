#!/usr/bin/env python3
"""
修复版本：在train_cov_dfp_3d.py中集成PrototypeMemory的示例
完全解决autograd图交叉引用问题
"""

import torch
from myutils.prototype_separation import PrototypeMemory


def train_epoch_with_prototype_separation_fixed(
    model, 
    train_loader, 
    optimizer, 
    proto_memory, 
    current_epoch, 
    args
):
    """
    修复版本的训练epoch函数，集成原型分离损失
    
    Args:
        model: 训练模型
        train_loader: 数据加载器
        optimizer: 优化器
        proto_memory: PrototypeMemory实例
        current_epoch: 当前epoch
        args: 训练参数
    """
    model.train()
    
    for batch_idx, (sampled_batch, is_labelled_mask) in enumerate(train_loader):
        # 提取数据
        volume_batch = sampled_batch['image']
        label_batch = sampled_batch['label']
        
        # **第1步**: 模型前向传播
        outputs_v, outputs_a, embedding_v, embedding_a = model(volume_batch)
        
        # **第2步**: 计算基础损失
        dice_loss = compute_dice_loss(outputs_v, label_batch)
        consistency_loss = compute_consistency_loss(outputs_v, outputs_a)
        hcc_loss = compute_hcc_loss(embedding_v, embedding_a)
        
        # **第3步**: 原型分离损失 - 关键修复点
        # 使用embedding_v作为特征，确保其requires_grad=True
        decoder_features = embedding_v  # (B, C, H, W, D)
        predictions = torch.softmax(outputs_v, dim=1)  # (B, K, H, W, D)
        
        # 计算原型损失（不更新原型，避免autograd图问题）
        proto_losses = proto_memory(
            feat=decoder_features,
            label=label_batch,
            pred=predictions,
            is_labelled=is_labelled_mask,
            epoch_idx=None  # 关键：设为None避免原型更新时的梯度问题
        )
        
        # **第4步**: 组合总损失
        total_loss = (
            args.lamda * dice_loss + 
            args.lambda_consistency * consistency_loss + 
            args.lambda_hcc * hcc_loss +
            args.lambda_prototype * proto_losses['total']  # 添加原型损失
        )
        
        # **第5步**: 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # **第6步**: 原型更新 - 在optimizer.step()之后独立进行
        if batch_idx % args.prototype_update_interval == 0:
            with torch.no_grad():
                # 创建无梯度的特征副本用于原型更新
                update_features = embedding_v.detach().clone()
                update_predictions = predictions.detach().clone()
                
                # 独立更新原型
                _ = proto_memory(
                    feat=update_features,
                    label=label_batch,
                    pred=update_predictions,
                    is_labelled=is_labelled_mask,
                    epoch_idx=current_epoch
                )
                
                # 清理临时张量
                del update_features, update_predictions
        
        # **第7步**: 日志记录（安全地提取损失值）
        if batch_idx % args.log_interval == 0:
            dice_val = dice_loss.detach().item()
            consistency_val = consistency_loss.detach().item()
            hcc_val = hcc_loss.detach().item()
            proto_intra_val = proto_losses['intra'].detach().item()
            proto_inter_val = proto_losses['inter'].detach().item()
            proto_total_val = proto_losses['total'].detach().item()
            total_val = total_loss.detach().item()
            
            print(f"Epoch {current_epoch}, Batch {batch_idx}:")
            print(f"  Dice: {dice_val:.4f}")
            print(f"  Consistency: {consistency_val:.4f}")
            print(f"  HCC: {hcc_val:.4f}")
            print(f"  Proto-Intra: {proto_intra_val:.4f}")
            print(f"  Proto-Inter: {proto_inter_val:.4f}")
            print(f"  Proto-Total: {proto_total_val:.4f}")
            print(f"  Total: {total_val:.4f}")


def initialize_prototype_memory_for_la_dataset(device):
    """为LA数据集初始化PrototypeMemory"""
    return PrototypeMemory(
        num_classes=1,          # LA数据集：1个前景类（心房）+ 1个背景类
        feat_dim=64,            # 根据你的embedding_dim设置
        proto_momentum=0.95,    # 高动量确保稳定更新
        conf_thresh=0.85,       # 高置信度阈值
        lambda_intra=0.3,       # 类内紧致性权重
        lambda_inter=0.1,       # 类间分离权重
        margin_m=1.5,           # 分离边界
        device=device
    ).to(device)


def main_training_example():
    """完整的训练示例"""
    print("LA数据集训练示例 - 集成原型分离模块")
    print("="*60)
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 初始化原型内存
    proto_memory = initialize_prototype_memory_for_la_dataset(device)
    print("✓ PrototypeMemory初始化完成")
    
    # 模拟训练参数
    class Args:
        lamda = 0.5
        lambda_consistency = 0.1
        lambda_hcc = 0.3
        lambda_prototype = 0.3        # 原型损失权重
        prototype_update_interval = 5  # 每5个batch更新一次原型
        log_interval = 10
    
    args = Args()
    
    print(f"\n训练参数:")
    print(f"  Dice损失权重: {args.lamda}")
    print(f"  一致性损失权重: {args.lambda_consistency}")
    print(f"  HCC损失权重: {args.lambda_hcc}")
    print(f"  原型损失权重: {args.lambda_prototype}")
    print(f"  原型更新间隔: {args.prototype_update_interval} batches")
    
    print(f"\n集成代码示例:")
    print("```python")
    print("# 在train_cov_dfp_3d.py的主训练循环中:")
    print("proto_memory = initialize_prototype_memory_for_la_dataset(device)")
    print("")
    print("for epoch in range(max_epochs):")
    print("    train_epoch_with_prototype_separation_fixed(")
    print("        model=model,")
    print("        train_loader=train_loader,")
    print("        optimizer=optimizer,")
    print("        proto_memory=proto_memory,")
    print("        current_epoch=epoch,")
    print("        args=args")
    print("    )")
    print("```")
    
    print(f"\n关键修复点:")
    print("1. ✓ 原型更新和损失计算完全分离")
    print("2. ✓ 使用detach().clone()创建无梯度副本")
    print("3. ✓ 在optimizer.step()后独立更新原型")
    print("4. ✓ 所有打印使用.detach().item()提取值")
    print("5. ✓ 及时清理临时张量避免内存泄漏")
    
    return proto_memory


# 辅助函数（模拟）
def compute_dice_loss(outputs, labels):
    """模拟Dice损失计算"""
    return torch.randn(1, requires_grad=True)

def compute_consistency_loss(outputs_v, outputs_a):
    """模拟一致性损失计算"""
    return torch.randn(1, requires_grad=True)

def compute_hcc_loss(embedding_v, embedding_a):
    """模拟HCC损失计算"""
    return torch.randn(1, requires_grad=True)


if __name__ == "__main__":
    print("完全修复的PrototypeMemory集成示例")
    print("="*60)
    
    try:
        proto_memory = main_training_example()
        
        print(f"\n✅ 集成示例创建成功！")
        print(f"💡 请将上述代码集成到您的train_cov_dfp_3d.py中")
        print(f"🔧 记住设置args.lambda_prototype=0.3作为原型损失权重")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc() 