#!/usr/bin/env python3
# 基于协方差的动态特征池(Cov-DFP)三阶段训练脚本

import os
import sys
import shutil
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import wandb

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from myutils import ramps, losses, test_patch
from myutils.dynamic_feature_pool import DynamicFeaturePool
from dataloaders.dataset import *
from networks.net_factory import net_factory
from myutils.cov_dynamic_feature_pool import CovarianceDynamicFeaturePool

def get_lambda_c(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_lambda_d(epoch):
    return args.consistency_o * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen

parser = argparse.ArgumentParser()
# 基础参数
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Root path of the project')
parser.add_argument('--dataset_path', type=str, default='/home/jovyan/work/medical_dataset/LA', help='Path to the dataset')
parser.add_argument('--exp', type=str, default='cov_dfp', help='exp_name')
parser.add_argument('--model', type=str, default='corn', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='total batch size')
parser.add_argument('--base_lr', type=float, default=0.01, help='base learning rate')
parser.add_argument('--labelnum', type=int, default=4, help='number of labeled samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
# 损失函数参数
parser.add_argument('--lamda', type=float, default=0.5, help='weight for supervised loss')
parser.add_argument('--consistency', type=float, default=1, help='consistency weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency rampup')
parser.add_argument('--temperature', type=float, default=0.4, help='temperature for sharpening')
# HCC参数
parser.add_argument('--hcc_weights', type=str, default='0.5,0.5,1,1,1.5', help='HCC layer weights')
parser.add_argument('--cov_mode', type=str, default='patch', choices=['full', 'patch'], help='covariance mode')
parser.add_argument('--patch_size', type=int, default=4, help='patch size for covariance')
# 其他参数
parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
parser.add_argument('--wandb_project', type=str, default='Cov-DFP', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')

args = parser.parse_args()

# Parse HCC weights
# hcc_weights = parse_hcc_weights(args.hcc_weights, num_layers=5)

snapshot_path = "./model/LA_{}_{}_dfp{}_memory{}_feat{}_compact{}_separate{}_proto{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}".format(
    args.exp,
    args.labelnum,
    args.num_dfp,
    args.memory_num,
    args.embedding_dim,
    args.lambda_compact,
    args.lambda_separate,
    args.lambda_prototype if args.use_prototype_separation else 0,
    args.num_filtered,
    args.consistency,
    args.consistency_rampup,
    args.consistency_o,
    args.max_iteration,
    args.seed)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.max_samples = 80
train_data_path = args.dataset_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

def train_stage_one(model, sampled_batch, optimizer, consistency_criterion, dice_loss, 
                   cov_dfp, proto_memory, iter_num, writer=None):
    """阶段一：初始预训练
    
    目标：训练初步的主模型，建立全局特征池，初始化原型
    """
    model.train()
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

    outputs_v, outputs_a, embedding_v, embedding_a, features_v, features_a = model(volume_batch, with_hcc=True)
    outputs_list = [outputs_v, outputs_a]
    num_outputs = len(outputs_list)

    # 确保张量在正确的设备上（GPU）
    device = volume_batch.device
    y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)
    y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)

    loss_s = 0
    for i in range(num_outputs):
        y = outputs_list[i][:labeled_bs, ...]
        y_prob = F.softmax(y, dim=1)
        loss_s += dice_loss(y_prob[:, 1, ...], label_batch[:labeled_bs, ...] == 1)

        y_all = outputs_list[i]
        y_prob_all = F.softmax(y_all, dim=1)
        y_ori[i] = y_prob_all
        y_pseudo_label[i] = sharpening(y_prob_all)

    loss_c = 0
    for i in range(num_outputs):
        for j in range(num_outputs):
            if i != j:
                loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

    # HCC Loss
    # loss_hcc = hierarchical_coral(features_v, features_a, hcc_weights,
    #                             cov_mode=args.cov_mode, 
    #                             patch_size=args.patch_size,
    #                             patch_strategy=args.hcc_patch_strategy,
    #                             topk=args.hcc_topk,
    #                             metric=args.hcc_metric,
    #                             scale=args.hcc_scale)

    # 原型分离损失（新增）
    loss_proto_intra = torch.tensor(0.0, device=device)
    loss_proto_inter = torch.tensor(0.0, device=device)
    loss_proto_total = torch.tensor(0.0, device=device)
    
    if args.use_prototype_separation and proto_memory is not None:
        # 使用embedding_v作为特征
        decoder_features = embedding_v  # (B, C, H, W, D)
        predictions = torch.softmax(outputs_v, dim=1)  # (B, K, H, W, D)
        
        # 创建is_labelled掩码
        is_labelled = torch.zeros(volume_batch.shape[0], dtype=torch.bool, device=device)
        is_labelled[:args.labeled_bs] = True
        
        # 计算原型损失（不更新原型，避免autograd图问题）
        proto_losses = proto_memory(
            feat=decoder_features,
            label=label_batch,
            pred=predictions,
            is_labelled=is_labelled,
            epoch_idx=None  # 关键：设为None避免原型更新时的梯度问题
        )
        
        loss_proto_intra = proto_losses['intra']
        loss_proto_inter = proto_losses['inter']
        loss_proto_total = proto_losses['total']

    lambda_c = get_lambda_c(iter_num // 150)
    total_loss = (args.lamda * loss_s + 
                  lambda_c * loss_c + 
                  args.lambda_prototype * loss_proto_total)

    # 添加特征到全局池（包含标签信息用于Two-Phase k-means）
    if args.use_dfp and cov_dfp is not None:
        # 获取标记样本的特征和标签
        labeled_features_v = embedding_v[:args.labeled_bs, ...]
        labeled_features_a = embedding_a[:args.labeled_bs, ...]
        labeled_labels = label_batch[:args.labeled_bs, ...]  # [labeled_bs, H, W, D]
        
        # 转换格式：[batch, h, w, d, feat_dim]
        labeled_features_v = labeled_features_v.permute(0, 2, 3, 4, 1).contiguous()
        labeled_features_a = labeled_features_a.permute(0, 2, 3, 4, 1).contiguous()
        
        # 投影到特征空间
        model.eval()
        proj_labeled_features_v = model.projection_head1(labeled_features_v.view(-1, labeled_features_v.shape[-1]))
        proj_labeled_features_a = model.projection_head2(labeled_features_a.view(-1, labeled_features_a.shape[-1]))
        model.train()
        
        # 平均两个分支的特征
        combined_features = (proj_labeled_features_v + proj_labeled_features_a) / 2
        
        # 准备对应的标签（展平到与特征相同的维度）
        flattened_labels = labeled_labels.view(-1)  # [labeled_bs * H * W * D]
        
        # 只保留有效的像素位置（非背景，假设0是背景）
        # 或者使用高置信度的预测作为伪标签
        with torch.no_grad():
            pred_probs = torch.softmax(outputs_v[:args.labeled_bs], dim=1)  # [labeled_bs, num_classes, H, W, D]
            pred_confidence = torch.max(pred_probs, dim=1)[0]  # [labeled_bs, H, W, D]
            pred_labels = torch.argmax(pred_probs, dim=1)  # [labeled_bs, H, W, D]
            
            # 使用高置信度区域 (置信度 > 0.9)
            high_conf_mask = (pred_confidence > 0.9).view(-1)  # [labeled_bs * H * W * D]
            
            if high_conf_mask.any():
                # 使用高置信度的预测作为标签
                final_features = combined_features[high_conf_mask]
                final_labels = pred_labels.view(-1)[high_conf_mask]
                
                # 添加到全局特征池（包含标签）
                cov_dfp.add_to_global_pool(final_features, final_labels)
            else:
                # 如果没有高置信度预测，使用真实标签
                cov_dfp.add_to_global_pool(combined_features, flattened_labels)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 原型更新 - 在optimizer.step()之后独立进行（新增）
    if args.use_prototype_separation and proto_memory is not None and iter_num % args.proto_update_interval == 0:
        with torch.no_grad():
            # 创建无梯度的特征副本用于原型更新
            update_features = embedding_v.detach().clone()
            update_predictions = torch.softmax(outputs_v, dim=1).detach().clone()
            
            # 独立更新原型
            _ = proto_memory(
                feat=update_features,
                label=label_batch,
                pred=update_predictions,
                is_labelled=is_labelled,
                epoch_idx=iter_num // 150  # 使用epoch作为更新索引
            )
            
            # 清理临时张量
            del update_features, update_predictions

    # 记录日志
    logging.info('Stage 1 - Iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_proto: %03f' % (
        iter_num, total_loss, loss_s, loss_c, loss_proto_total))
    
    return {
        'total_loss': total_loss.item(),
        'loss_s': loss_s.item(),
        'loss_c': loss_c.item(),
        'loss_proto_intra': loss_proto_intra.item(),
        'loss_proto_inter': loss_proto_inter.item(),
        'loss_proto_total': loss_proto_total.item(),
        'lambda_c': lambda_c
    }

def train_stage_two_build_dfp(cov_dfp, max_optimization_iterations=50, learning_rate=0.01):
    """阶段二：构建DFP并生成Selector训练目标 - 度量学习驱动"""
    logging.info(f"Building DFPs with metric learning: max_iter={max_optimization_iterations}, lr={learning_rate}")
    success = cov_dfp.build_dfps(max_optimization_iterations=max_optimization_iterations, 
                                learning_rate=learning_rate)
    if success:
        stats = cov_dfp.get_statistics()
        logging.info(f"Metric-learning driven DFP construction successful: {stats}")
        return True
    else:
        logging.warning("DFP construction failed")
        return False

def train_stage_three_selector(model, sampled_batch, selector_optimizer, cov_dfp, iter_num):
    """阶段三A：训练Selector"""
    model.eval()  # 冻结主模型
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()
    
    with torch.no_grad():
        outputs_v, outputs_a, embedding_v, embedding_a = model(volume_batch)
        
        # 获取区域特征
        labeled_features_v = embedding_v[:args.labeled_bs, ...]
        labeled_features_a = embedding_a[:args.labeled_bs, ...]
        
        # 转换格式
        labeled_features_v = labeled_features_v.permute(0, 2, 3, 4, 1).contiguous()
        labeled_features_a = labeled_features_a.permute(0, 2, 3, 4, 1).contiguous()
        
        # 投影特征
        proj_labeled_features_v = model.projection_head1(labeled_features_v.view(-1, labeled_features_v.shape[-1]))
        proj_labeled_features_a = model.projection_head2(labeled_features_a.view(-1, labeled_features_a.shape[-1]))
        
        # 合并特征
        combined_features = (proj_labeled_features_v + proj_labeled_features_a) / 2
    
    # 生成目标标签
    target_labels = cov_dfp.get_dfp_target_labels(combined_features)
    
    # 训练Selector
    model.train()
    selector_logits = model.dfp_selector(combined_features)
    selector_loss = F.cross_entropy(selector_logits, target_labels)
    
    selector_optimizer.zero_grad()
    selector_loss.backward()
    selector_optimizer.step()
    
    # 计算准确率
    with torch.no_grad():
        predicted = torch.argmax(selector_logits, dim=1)
        accuracy = (predicted == target_labels).float().mean()
    
    logging.info(f'Stage 3A - Selector training iter {iter_num}: loss={selector_loss.item():.4f}, acc={accuracy.item():.4f}')
    
    return {
        'selector_loss': selector_loss.item(),
        'selector_accuracy': accuracy.item()
    }

def train_stage_three_main(model, sampled_batch, optimizer, consistency_criterion, dice_loss, 
                          cov_dfp, proto_memory, iter_num, writer=None):
    """阶段三B：主模型训练（使用Selector选择的DFP + 原型分离）"""
    model.train()
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

    # 获取区域特征用于Selector
    with torch.no_grad():
        temp_outputs_v, temp_outputs_a, temp_embedding_v, temp_embedding_a = model(volume_batch)
        
        labeled_features_v = temp_embedding_v[:args.labeled_bs, ...]
        labeled_features_a = temp_embedding_a[:args.labeled_bs, ...]
        
        labeled_features_v = labeled_features_v.permute(0, 2, 3, 4, 1).contiguous()
        labeled_features_a = labeled_features_a.permute(0, 2, 3, 4, 1).contiguous()
        
        proj_labeled_features_v = model.projection_head1(labeled_features_v.view(-1, labeled_features_v.shape[-1]))
        proj_labeled_features_a = model.projection_head2(labeled_features_a.view(-1, labeled_features_a.shape[-1]))
        
        combined_features = (proj_labeled_features_v + proj_labeled_features_a) / 2
        dfp_predictions = model.dfp_selector.predict_dfp(combined_features)

    # 正常前向传播
    outputs_v, outputs_a, embedding_v, embedding_a, features_v, features_a = model(volume_batch, with_hcc=True)
    outputs_list = [outputs_v, outputs_a]
    num_outputs = len(outputs_list)

    # 确保张量在正确的设备上（GPU）
    device = volume_batch.device
    y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)
    y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)

    loss_s = 0
    for i in range(num_outputs):
        y = outputs_list[i][:labeled_bs, ...]
        y_prob = F.softmax(y, dim=1)
        loss_s += dice_loss(y_prob[:, 1, ...], label_batch[:labeled_bs, ...] == 1)

        y_all = outputs_list[i]
        y_prob_all = F.softmax(y_all, dim=1)
        y_ori[i] = y_prob_all
        y_pseudo_label[i] = sharpening(y_prob_all)

    loss_c = 0
    for i in range(num_outputs):
        for j in range(num_outputs):
            if i != j:
                loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

    # HCC Loss
    # loss_hcc = hierarchical_coral(features_v, features_a, hcc_weights,
    #                             cov_mode=args.cov_mode, 
    #                             patch_size=args.patch_size,
    #                             patch_strategy=args.hcc_patch_strategy,
    #                             topk=args.hcc_topk,
    #                             metric=args.hcc_metric,
    #                             scale=args.hcc_scale)
    loss_hcc = torch.tensor(0.0, device=device)  # 临时设置为0

    # 原型分离损失（新增）
    loss_proto_intra = torch.tensor(0.0, device=device)
    loss_proto_inter = torch.tensor(0.0, device=device)
    loss_proto_total = torch.tensor(0.0, device=device)
    
    if args.use_prototype_separation and proto_memory is not None:
        # 使用embedding_v作为特征
        decoder_features = embedding_v  # (B, C, H, W, D)
        predictions = torch.softmax(outputs_v, dim=1)  # (B, K, H, W, D)
        
        # 创建is_labelled掩码
        is_labelled = torch.zeros(volume_batch.shape[0], dtype=torch.bool, device=device)
        is_labelled[:args.labeled_bs] = True
        
        # 计算原型损失（不更新原型，避免autograd图问题）
        proto_losses = proto_memory(
            feat=decoder_features,
            label=label_batch,
            pred=predictions,
            is_labelled=is_labelled,
            epoch_idx=None  # 关键：设为None避免原型更新时的梯度问题
        )
        
        loss_proto_intra = proto_losses['intra']
        loss_proto_inter = proto_losses['inter']
        loss_proto_total = proto_losses['total']

    # 度量学习损失 (新增) - 修复初始化问题
    loss_compact = torch.tensor(0.0, device=device)
    loss_separate = torch.tensor(0.0, device=device)
    
    if args.use_dfp and cov_dfp is not None and cov_dfp.dfps_built:
        # 获取区域特征用于度量学习损失计算
        region_features_v = embedding_v[:args.labeled_bs, ...]
        region_features_a = embedding_a[:args.labeled_bs, ...]
        
        # 转换格式并投影
        region_features_v = region_features_v.permute(0, 2, 3, 4, 1).contiguous()
        region_features_a = region_features_a.permute(0, 2, 3, 4, 1).contiguous()
        
        # 获取投影特征（需要暂时切换到eval模式）
        model.eval()
        proj_region_features_v = model.projection_head1(region_features_v.view(-1, region_features_v.shape[-1]))
        proj_region_features_a = model.projection_head2(region_features_a.view(-1, region_features_a.shape[-1]))
        model.train()
        
        # 平均两个分支的特征
        combined_region_features = (proj_region_features_v + proj_region_features_a) / 2  # [N, D]
        
        # 使用Selector预测DFP分配
        with torch.no_grad():
            dfp_predictions = model.dfp_selector.predict_dfp(combined_region_features)  # [N]
        
        # 按DFP分组特征
        batch_features_by_dfp = cov_dfp.group_features_by_dfp_predictions(
            combined_region_features, dfp_predictions
        )
        
        # 计算度量学习损失
        loss_compact, loss_separate = cov_dfp.compute_metric_learning_losses(
            batch_features_by_dfp, margin=args.separation_margin
        )

    lambda_c = get_lambda_c(iter_num // 150)
    total_loss = (args.lamda * loss_s + 
                  lambda_c * loss_c + 
                  args.lambda_hcc * loss_hcc + 
                  args.lambda_compact * loss_compact + 
                  args.lambda_separate * loss_separate +
                  args.lambda_prototype * loss_proto_total)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 在反向传播后更新DFPs
    if args.use_dfp and cov_dfp is not None and cov_dfp.dfps_built:
        with torch.no_grad():
            cov_dfp.update_dfps_with_batch_features(batch_features_by_dfp, 
                                                   update_rate=0.1, 
                                                   max_dfp_size=1000)

    # 原型更新 - 在optimizer.step()之后独立进行（新增）
    if args.use_prototype_separation and proto_memory is not None and iter_num % args.proto_update_interval == 0:
        with torch.no_grad():
            # 创建无梯度的特征副本用于原型更新
            update_features = embedding_v.detach().clone()
            update_predictions = torch.softmax(outputs_v, dim=1).detach().clone()
            
            # 独立更新原型
            _ = proto_memory(
                feat=update_features,
                label=label_batch,
                pred=update_predictions,
                is_labelled=is_labelled,
                epoch_idx=iter_num // 150  # 使用epoch作为更新索引
            )
            
            # 清理临时张量
            del update_features, update_predictions

    logging.info('Stage 3B - Main training iter %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_hcc: %03f, loss_compact: %03f, loss_separate: %03f, loss_proto: %03f' % (
        iter_num, total_loss, loss_s, loss_c, loss_hcc, loss_compact, loss_separate, loss_proto_total))
    
    return {
        'total_loss': total_loss.item(),
        'loss_s': loss_s.item(),
        'loss_c': loss_c.item(),
        'loss_hcc': loss_hcc.item(),
        'loss_compact': loss_compact.item(),
        'loss_separate': loss_separate.item(),
        'loss_proto_intra': loss_proto_intra.item(),
        'loss_proto_inter': loss_proto_inter.item(),
        'loss_proto_total': loss_proto_total.item(),
        'lambda_c': lambda_c
    }

if __name__ == "__main__":
    # 创建保存目录
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp}_{args.model}_dfp{args.num_dfp}_feat{args.embedding_dim}_compact{args.lambda_compact}_separate{args.lambda_separate}_proto{args.lambda_prototype if args.use_prototype_separation else 0}",
        )
        logging.info(f"Wandb initialized with project: {args.wandb_project}")

    # 创建模型
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("Model moved to GPU")
    else:
        logging.warning("CUDA not available, using CPU")

    # 创建CovarianceDynamicFeaturePool
    feature_pool = CovarianceDynamicFeaturePool(feature_dim=32, num_dfp=2, max_global_features=10000, device='cuda')

    # 创建数据加载器
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),
                           with_idx=True)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            iter_num += 1
            model.train()
            volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs, embedding = model(volume_batch)
            # 只用有标签部分
            labeled_features = embedding[:args.labeled_bs, ...]
            labeled_labels = label_batch[:args.labeled_bs, ...]
            labeled_features = labeled_features.permute(0, 2, 3, 4, 1).contiguous().view(-1, embedding.shape[1])
            labeled_labels = labeled_labels.view(-1)
            feature_pool.add_to_global_pool(labeled_features, labeled_labels)
            # 损失与优化
            # 以outputs为二分类输出，取前labeled_bs个batch
            y = outputs[:args.labeled_bs, ...]
            y_prob = torch.softmax(y, dim=1)
            loss = dice_loss(y_prob[:, 1, ...], label_batch[:args.labeled_bs, ...] == 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 日志
            logging.info(f"Iter {iter_num}: loss={loss:.4f}")
            # 验证与保存
            if iter_num >= 1000 and iter_num % 500 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA', 
                                                          dataset_path=args.dataset_path)
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
                    save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info(f"save best model to {save_mode_path}")
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()
            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save model to {save_mode_path}")
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    if args.use_wandb:
        wandb.finish() 