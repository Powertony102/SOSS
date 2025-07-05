#!/usr/bin/env python3
"""
集成示例：在现有的3D半监督分割训练框架中集成PrototypeMemory模块

此文件展示了如何将Inter-Class Prototype Separation Module集成到您现有的
train_cov_dfp_3d.py训练脚本中。
"""

import torch
import torch.nn.functional as F
import logging
from myutils.prototype_separation import PrototypeMemory


def create_prototype_memory(args):
    """
    创建PrototypeMemory实例，配置适合您当前框架的参数
    
    Args:
        args: 训练脚本的参数配置
        
    Returns:
        PrototypeMemory: 配置好的原型内存模块
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 针对LA数据集的配置 (二分类：背景+左心房)
    num_classes = 1  # 只有左心房这一个前景类别
    
    # 如果是多类别数据集，根据实际情况调整
    if hasattr(args, 'num_classes'):
        num_classes = args.num_classes - 1  # 减去背景类
    
    proto_mem = PrototypeMemory(
        num_classes=num_classes,
        feat_dim=args.embedding_dim,  # 与您的网络特征维度一致
        proto_momentum=0.95,          # 较高的动量保证稳定性
        conf_thresh=0.85,             # 高置信度阈值过滤噪声
        update_interval=1,            # 每个epoch都更新
        lambda_intra=0.3,             # 类内紧致性权重 (可调节)
        lambda_inter=0.1,             # 类间分离权重 (可调节)
        margin_m=1.5,                 # 分离边界
        device=device
    ).to(device)
    
    return proto_mem


def train_stage_with_prototype_separation(
    model, sampled_batch, optimizer, consistency_criterion, dice_loss, 
    cov_dfp, proto_memory, iter_num, args, writer=None
):
    """
    修改后的训练函数，集成了PrototypeMemory模块
    
    这是对原始train_stage_one/train_stage_three_main函数的增强版本
    """
    model.train()
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

    # 获取模型输出和特征
    outputs_v, outputs_a, embedding_v, embedding_a, features_v, features_a = model(volume_batch, with_hcc=True)
    outputs_list = [outputs_v, outputs_a]
    num_outputs = len(outputs_list)

    # 确保张量在正确的设备上
    device = volume_batch.device
    y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)
    y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)

    # 计算标准损失
    loss_s = 0
    for i in range(num_outputs):
        y = outputs_list[i][:args.labeled_bs, ...]
        y_prob = F.softmax(y, dim=1)
        loss_s += dice_loss(y_prob[:, 1, ...], label_batch[:args.labeled_bs, ...] == 1)

        y_all = outputs_list[i]
        y_prob_all = F.softmax(y_all, dim=1)
        y_ori[i] = y_prob_all
        y_pseudo_label[i] = sharpening(y_prob_all, args.temperature)

    # 一致性损失
    loss_c = 0
    for i in range(num_outputs):
        for j in range(num_outputs):
            if i != j:
                loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

    # ======================== 新增：原型分离损失 ========================
    # 使用主分支的特征和预测进行原型学习
    decoder_features = embedding_v  # (B, C, H, W, D)
    predictions = torch.softmax(outputs_v, dim=1)  # (B, K, H, W, D)
    
    # 创建标签掩码：前labeled_bs个样本有标签，后面的是无标签样本
    batch_size = volume_batch.shape[0]
    is_labelled = torch.zeros(batch_size, dtype=torch.bool, device=device)
    is_labelled[:args.labeled_bs] = True
    
    # 当前epoch (从iter_num估算)
    current_epoch = iter_num // 150  # 假设每个epoch有150个iteration
    
    # 计算原型分离损失
    proto_losses = proto_memory(
        feat=decoder_features,
        label=label_batch,
        pred=predictions,
        is_labelled=is_labelled,
        epoch_idx=current_epoch
    )
    
    loss_prototype = proto_losses['total']
    # ================================================================

    # 获取动态权重
    lambda_c = get_lambda_c(current_epoch, args)
    
    # 合并所有损失
    total_loss = (args.lamda * loss_s + 
                  lambda_c * loss_c + 
                  args.lambda_prototype * loss_prototype)  # 新增原型损失项

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 记录详细日志
    logging.info(
        'Iter %d: total=%.4f, supervised=%.4f, consistency=%.4f, '
        'prototype=%.4f (intra=%.4f, inter=%.4f), confident_pixels=%d' % (
            iter_num, total_loss.item(), loss_s.item(), loss_c.item(),
            loss_prototype.item(), proto_losses['intra'].item(), 
            proto_losses['inter'].item(), proto_losses['n_confident_pixels']
        )
    )
    
    return {
        'total_loss': total_loss.item(),
        'loss_s': loss_s.item(),
        'loss_c': loss_c.item(),
        'loss_prototype': loss_prototype.item(),
        'proto_intra': proto_losses['intra'].item(),
        'proto_inter': proto_losses['inter'].item(),
        'lambda_c': lambda_c,
        'confident_pixels': proto_losses['n_confident_pixels'],
        'initialized_protos': proto_losses['n_initialized_protos']
    }


def add_prototype_args(parser):
    """
    为argparse添加原型分离相关的参数
    
    Args:
        parser: argparse.ArgumentParser实例
    """
    # 原型分离参数组
    proto_group = parser.add_argument_group('Prototype Separation', 'Inter-class prototype separation parameters')
    
    proto_group.add_argument('--use_prototype', action='store_true', 
                           help='enable inter-class prototype separation')
    proto_group.add_argument('--lambda_prototype', type=float, default=0.3,
                           help='weight for prototype separation loss')
    proto_group.add_argument('--proto_momentum', type=float, default=0.95,
                           help='momentum for prototype updates')
    proto_group.add_argument('--proto_conf_thresh', type=float, default=0.85,
                           help='confidence threshold for prototype updates')
    proto_group.add_argument('--proto_lambda_intra', type=float, default=1.0,
                           help='weight for intra-class compactness loss')
    proto_group.add_argument('--proto_lambda_inter', type=float, default=0.1,
                           help='weight for inter-class separation loss')
    proto_group.add_argument('--proto_margin', type=float, default=1.5,
                           help='margin for inter-class separation')
    proto_group.add_argument('--proto_update_interval', type=int, default=1,
                           help='update prototypes every N epochs')


def get_lambda_c(epoch, args):
    """获取一致性损失权重 (从原始代码移植)"""
    from myutils import ramps
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P, temperature):
    """预测锐化函数 (从原始代码移植)"""
    T = 1 / temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


# ======================== 集成指南 ========================

def integration_guide():
    """
    详细的集成指南，说明如何修改现有的train_cov_dfp_3d.py
    """
    
    guide = """
    ==================== PrototypeMemory 集成指南 ====================
    
    1. 导入模块
    在 train_cov_dfp_3d.py 的顶部添加：
    ```python
    from myutils.prototype_separation import PrototypeMemory
    ```
    
    2. 添加命令行参数
    在参数解析部分添加：
    ```python
    # 原型分离参数
    parser.add_argument('--use_prototype', action='store_true', 
                       help='enable inter-class prototype separation')
    parser.add_argument('--lambda_prototype', type=float, default=0.3,
                       help='weight for prototype separation loss')
    parser.add_argument('--proto_momentum', type=float, default=0.95,
                       help='momentum for prototype updates')
    parser.add_argument('--proto_conf_thresh', type=float, default=0.85,
                       help='confidence threshold for prototype updates')
    ```
    
    3. 初始化PrototypeMemory
    在main函数中，创建模型后添加：
    ```python
    # 创建原型内存模块
    proto_memory = None
    if args.use_prototype:
        proto_memory = PrototypeMemory(
            num_classes=1,  # LA数据集只有一个前景类
            feat_dim=args.embedding_dim,
            proto_momentum=args.proto_momentum,
            conf_thresh=args.proto_conf_thresh,
            lambda_intra=1.0,
            lambda_inter=0.1,
            margin_m=1.5,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("PrototypeMemory initialized")
    ```
    
    4. 修改训练函数
    在各个训练阶段函数中添加原型损失计算：
    ```python
    # 在现有损失计算后添加
    loss_prototype = torch.tensor(0.0, device=device)
    if args.use_prototype and proto_memory is not None:
        decoder_features = embedding_v  # 或者其他合适的特征
        predictions = torch.softmax(outputs_v, dim=1)
        
        # 创建标签掩码
        is_labelled = torch.zeros(batch_size, dtype=torch.bool, device=device)
        is_labelled[:args.labeled_bs] = True
        
        # 计算原型损失
        proto_losses = proto_memory(
            feat=decoder_features,
            label=label_batch,
            pred=predictions,
            is_labelled=is_labelled,
            epoch_idx=iter_num // 150
        )
        loss_prototype = proto_losses['total']
    
    # 更新总损失
    total_loss = (args.lamda * loss_s + 
                  lambda_c * loss_c + 
                  args.lambda_hcc * loss_hcc +
                  args.lambda_prototype * loss_prototype)  # 新增
    ```
    
    5. 日志记录
    在日志记录部分添加原型损失信息：
    ```python
    if args.use_prototype and proto_memory is not None:
        stats = proto_memory.get_prototype_statistics()
        if args.use_wandb:
            wandb.log({
                'prototype/loss_total': loss_prototype.item(),
                'prototype/loss_intra': proto_losses['intra'].item(),
                'prototype/loss_inter': proto_losses['inter'].item(),
                'prototype/confident_pixels': proto_losses['n_confident_pixels'],
                'prototype/initialized_protos': proto_losses['n_initialized_protos'],
                'iteration': iter_num
            })
    ```
    
    6. 运行命令示例
    ```bash
    python train_cov_dfp_3d.py \\
        --use_prototype \\
        --lambda_prototype 0.3 \\
        --proto_momentum 0.95 \\
        --proto_conf_thresh 0.85 \\
        --other_existing_args...
    ```
    
    ==================== 关键注意事项 ====================
    
    1. 特征选择：
       - 使用 embedding_v 作为解码器特征是推荐的选择
       - 确保特征维度与 args.embedding_dim 一致
    
    2. 类别设置：
       - LA数据集：num_classes=1 (只有左心房前景类)
       - 其他数据集：根据实际前景类别数调整
    
    3. 权重调节：
       - lambda_prototype: 0.1-0.5 通常效果较好
       - proto_conf_thresh: 0.8-0.9 平衡质量和数量
       - proto_momentum: 0.9-0.99 保证稳定性
    
    4. 内存优化：
       - 模块会自动管理原型内存
       - 高置信度掩码减少计算开销
    
    5. 调试建议：
       - 使用 get_prototype_statistics() 监控原型状态
       - 观察 confident_pixels 数量确保有效更新
       - 监控 intra/inter 损失比例
    
    ==============================================================
    """
    
    return guide


if __name__ == "__main__":
    print(integration_guide()) 