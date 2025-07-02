#!/usr/bin/env python3
"""
基于协方差的动态特征池(Cov-DFP)训练脚本 - ACDC数据集适配版本
参考SS-Net的ACDC数据处理方式，结合Cov-DFP框架进行心脏图像分割
"""

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
import time
import math
import itertools
import warnings

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from myutils import ramps, losses, test_patch
from myutils.cov_dynamic_feature_pool import CovarianceDynamicFeaturePool
from myutils.hcc_loss import hierarchical_coral, parse_hcc_weights
from dataloaders.acdc_dataset import ACDCDataSet, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from dataloaders.data_utils import ACDCDataProcessor  # 导入数据处理工具
from myutils.covariance_utils import compute_covariance_matrix, coral_loss

warnings.filterwarnings("ignore", category=UserWarning)

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
parser.add_argument('--dataset_name', type=str, default='ACDC', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Root path of the project')
parser.add_argument('--dataset_path', type=str, default='/home/jovyan/work/medical_dataset/ACDC', help='Path to the ACDC dataset')
parser.add_argument('--exp', type=str, default='acdc_soss', help='exp_name')
parser.add_argument('--model', type=str, default='corn2d', help='model_name: unet, vnet, corn, corn2d')
parser.add_argument('--max_iteration', type=int, default=20000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=200, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='total batch size')
parser.add_argument('--base_lr', type=float, default=0.01, help='base learning rate')
parser.add_argument('--labelnum', type=int, default=7, help='number of labeled samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')

# ACDC特定参数
parser.add_argument('--patch_size', type=tuple, default=(256, 256), help='patch size for 2D ACDC')
parser.add_argument('--use_h5', action='store_true', default=True, help='use preprocessed H5 files')

# 损失函数参数
parser.add_argument('--lamda', type=float, default=0.5, help='weight for supervised loss')
parser.add_argument('--consistency', type=float, default=1, help='consistency weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency rampup')
parser.add_argument('--temperature', type=float, default=0.4, help='temperature for sharpening')
parser.add_argument('--lambda_hcc', type=float, default=0.1, help='weight for HCC loss')

# HCC参数
parser.add_argument('--hcc_weights', type=str, default='0.5,0.5,1,1,1.5', help='HCC layer weights')
parser.add_argument('--cov_mode', type=str, default='patch', choices=['full', 'patch'], help='covariance mode')
parser.add_argument('--hcc_patch_size', type=int, default=4, help='patch size for covariance')

# DFP参数
parser.add_argument('--use_dfp', action='store_true', help='whether to use dynamic feature pool')
parser.add_argument('--num_dfp', type=int, default=8, help='number of dynamic feature pools')
parser.add_argument('--dfp_start_iter', type=int, default=2000, help='iteration to start building DFPs')
parser.add_argument('--selector_train_iter', type=int, default=50, help='iterations for training selector')
parser.add_argument('--dfp_reconstruct_interval', type=int, default=1000, help='interval for reconstructing DFPs')
parser.add_argument('--max_global_features', type=int, default=50000, help='maximum global features')
parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dimension')

# 度量学习参数
parser.add_argument('--lambda_compact', type=float, default=0.1, help='weight for intra-pool compactness loss')
parser.add_argument('--lambda_separate', type=float, default=0.05, help='weight for inter-pool separation loss')
parser.add_argument('--separation_margin', type=float, default=1.0, help='margin for inter-pool separation loss')

# 其他参数
parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
parser.add_argument('--wandb_project', type=str, default='SOSS-ACDC', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
parser.add_argument('--consistency_o', type=float, default=0.05, help='lambda_s to balance sim loss')
parser.add_argument('--hcc_patch_strategy', type=str, default='mean_cov', 
                    choices=['mean_cov', 'mean_loss', 'max_loss', 'topk'],
                    help='patch strategy for HCC loss')
parser.add_argument('--hcc_topk', type=int, default=5, help='top-k value when hcc_patch_strategy is topk')
parser.add_argument('--hcc_metric', type=str, default='fro', choices=['fro', 'log'],
                    help='metric for HCC loss computation')
parser.add_argument('--hcc_scale', type=float, default=1.0, help='scaling factor for HCC loss')
parser.add_argument('--hcc_divide_by_dim', action='store_true', default=True, 
                    help='whether to divide CORAL loss by dimension squared')
parser.add_argument('--memory_num', type=int, default=256, help='num of embeddings per class in memory bank')
parser.add_argument('--num_filtered', type=int, default=12800,
                    help='num of unlabeled embeddings to calculate similarity')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')

args = parser.parse_args()

# Parse HCC weights
hcc_weights = parse_hcc_weights(args.hcc_weights, num_layers=5)

# ACDC特定配置
num_classes = 4  # ACDC: 背景, LV, RV, MYO
patch_size = args.patch_size

snapshot_path = "./model/ACDC_{}_{}_dfp{}_memory{}_feat{}_compact{}_separate{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
    args.exp,
    args.labelnum,
    args.num_dfp,
    args.memory_num,
    args.embedding_dim,
    args.lambda_compact,
    args.lambda_separate,
    args.num_filtered,
    args.consistency,
    args.consistency_rampup,
    args.consistency_o,
    args.max_iteration,
    args.seed,
    args.model)

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
                   cov_dfp, iter_num, writer=None, data_processor=None):
    """阶段一：初始预训练
    
    目标：训练初步的主模型，建立全局特征池
    """
    model.train()
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

    # 使用专门的数据处理器进行数据检查和修复
    if data_processor is not None:
        volume_batch, label_batch, validation = data_processor.process_batch(volume_batch, label_batch)
        if not validation['valid']:
            print(f"数据验证失败: {validation['errors']}")
    
    # 直接使用2D数据进行前向传播
    model_output = model(volume_batch, with_hcc=True)
    if isinstance(model_output, dict):
        outputs_v = model_output['seg1']
        outputs_a = model_output['seg2']
        embedding_v = model_output['embedding1']
        embedding_a = model_output['embedding2']
        features_v = model_output['features1']
        features_a = model_output['features2']
    else:
        outputs_v, outputs_a, embedding_v, embedding_a, features_v, features_a = model_output
        
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
        
        # ACDC多类别Dice损失
        loss_dice_class = 0
        for class_idx in range(1, num_classes):  # 跳过背景类
            loss_dice_class += dice_loss(y_prob[:, class_idx, ...], 
                                       (label_batch[:labeled_bs, ...] == class_idx).float())
        loss_s += loss_dice_class / (num_classes - 1)

        y_all = outputs_list[i]
        y_prob_all = F.softmax(y_all, dim=1)
        y_ori[i] = y_prob_all
        y_pseudo_label[i] = sharpening(y_prob_all)

    loss_c = 0
    for i in range(num_outputs):
        for j in range(num_outputs):
            if i != j:
                loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

    # HCC Loss - 适配2D特征
    loss_hcc = hierarchical_coral(features_v, features_a, hcc_weights,
                                cov_mode=args.cov_mode, 
                                patch_size=args.hcc_patch_size,
                                patch_strategy=args.hcc_patch_strategy,
                                topk=args.hcc_topk,
                                metric=args.hcc_metric,
                                scale=args.hcc_scale)

    lambda_c = get_lambda_c(iter_num // 150)
    total_loss = args.lamda * loss_s + lambda_c * loss_c + args.lambda_hcc * loss_hcc

    # 添加特征到全局池
    if args.use_dfp and cov_dfp is not None:
        # 获取正确预测的特征
        labeled_features_v = embedding_v[:args.labeled_bs, ...]
        labeled_features_a = embedding_a[:args.labeled_bs, ...]
        
        # 对于2D数据，调整特征维度 [batch, feat_dim, h, w] -> [batch, h, w, feat_dim]
        if len(labeled_features_v.shape) == 4:
            labeled_features_v = labeled_features_v.permute(0, 2, 3, 1).contiguous()
            labeled_features_a = labeled_features_a.permute(0, 2, 3, 1).contiguous()
        
        # 投影到特征空间
        model.eval()
        proj_labeled_features_v = model.projection_head1(labeled_features_v.view(-1, labeled_features_v.shape[-1]))
        proj_labeled_features_a = model.projection_head2(labeled_features_a.view(-1, labeled_features_a.shape[-1]))
        model.train()
        
        # 获取正确预测的掩码
        with torch.no_grad():
            pred_v = torch.argmax(torch.softmax(outputs_v[:labeled_bs], dim=1), dim=1)
            pred_a = torch.argmax(torch.softmax(outputs_a[:labeled_bs], dim=1), dim=1)
            correct_v = (pred_v == label_batch[:labeled_bs]).float()
            correct_a = (pred_a == label_batch[:labeled_bs]).float()
        
        # 添加正确预测的特征到全局池
        # 将两个特征合并后添加到全局池
        combined_features = torch.cat([
            proj_labeled_features_v.view(-1, args.embedding_dim),
            proj_labeled_features_a.view(-1, args.embedding_dim)
        ], dim=0)
        cov_dfp.add_to_global_pool(combined_features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 记录日志
    if writer:
        writer.add_scalar('train/loss_s', loss_s, iter_num)
        writer.add_scalar('train/loss_c', loss_c, iter_num)
        writer.add_scalar('train/loss_hcc', loss_hcc, iter_num)
        writer.add_scalar('train/total_loss', total_loss, iter_num)
        writer.add_scalar('train/lambda_c', lambda_c, iter_num)
    
    if args.use_wandb:
        wandb.log({
            'train/loss_s': loss_s.item(),
            'train/loss_c': loss_c.item(), 
            'train/loss_hcc': loss_hcc.item(),
            'train/total_loss': total_loss.item(),
            'train/lambda_c': lambda_c,
            'iteration': iter_num
        })

    return total_loss.item()

def train_stage_two(model, sampled_batch, optimizer, selector_optimizer, consistency_criterion, 
                   dice_loss, cov_dfp, iter_num, writer=None, data_processor=None):
    """阶段二：DFP构建与度量学习训练
    
    目标：构建动态特征池，训练选择器网络，执行度量学习
    """
    model.train()
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

    # 使用专门的数据处理器进行数据检查和修复
    if data_processor is not None:
        volume_batch, label_batch, validation = data_processor.process_batch(volume_batch, label_batch)
        if not validation['valid']:
            print(f"数据验证失败: {validation['errors']}")

    # 前向传播 - 直接使用2D数据
    model_output = model(volume_batch, with_hcc=True)
    if isinstance(model_output, dict):
        outputs_v = model_output['seg1']
        outputs_a = model_output['seg2']
        embedding_v = model_output['embedding1']
        embedding_a = model_output['embedding2']
        features_v = model_output['features1']
        features_a = model_output['features2']
    else:
        outputs_v, outputs_a, embedding_v, embedding_a, features_v, features_a = model_output
    
    # 构建DFPs
    if not cov_dfp.dfps_built and cov_dfp.get_global_pool_size() > args.num_dfp * 10:
        success = cov_dfp.build_dfps()
        if success:
            logging.info(f"Built DFPs at iteration {iter_num}")
        else:
            logging.warning(f"Failed to build DFPs at iteration {iter_num}")

    # 计算基础损失
    outputs_list = [outputs_v, outputs_a]
    loss_s = 0
    for i, outputs in enumerate(outputs_list):
        y = outputs[:labeled_bs, ...]
        y_prob = F.softmax(y, dim=1)
        
        # ACDC多类别Dice损失
        loss_dice_class = 0
        for class_idx in range(1, num_classes):
            loss_dice_class += dice_loss(y_prob[:, class_idx, ...], 
                                       (label_batch[:labeled_bs, ...] == class_idx).float())
        loss_s += loss_dice_class / (num_classes - 1)

    # 一致性损失
    y_ori = torch.stack([F.softmax(out, dim=1) for out in outputs_list])
    y_pseudo_label = torch.stack([sharpening(F.softmax(out, dim=1)) for out in outputs_list])
    
    loss_c = 0
    for i in range(len(outputs_list)):
        for j in range(len(outputs_list)):
            if i != j:
                loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

    # HCC损失
    loss_hcc = hierarchical_coral(features_v, features_a, hcc_weights,
                                cov_mode=args.cov_mode, 
                                patch_size=args.hcc_patch_size,
                                patch_strategy=args.hcc_patch_strategy,
                                topk=args.hcc_topk,
                                metric=args.hcc_metric,
                                scale=args.hcc_scale)

    # 度量学习损失
    loss_compact, loss_separate = torch.tensor(0.0, device='cuda'), torch.tensor(0.0, device='cuda')
    if cov_dfp.dfps_built:
        # 对于2D数据，调整特征维度
        if len(embedding_v.shape) == 4:
            embedding_v_reshaped = embedding_v.permute(0, 2, 3, 1).contiguous()
            embedding_a_reshaped = embedding_a.permute(0, 2, 3, 1).contiguous()
        
        proj_embedding_v = model.projection_head1(embedding_v_reshaped.view(-1, embedding_v_reshaped.shape[-1]))
        proj_embedding_a = model.projection_head2(embedding_a_reshaped.view(-1, embedding_a_reshaped.shape[-1]))
        
        proj_embedding_v = proj_embedding_v.view(*embedding_v_reshaped.shape[:-1], -1)
        proj_embedding_a = proj_embedding_a.view(*embedding_a_reshaped.shape[:-1], -1)
        
        # 计算度量学习损失
        # 首先需要将特征按DFP分组
        batch_features_by_dfp = {}
        # 这里需要根据实际的Selector预测来分组特征
        # 暂时使用简单的分组方式
        batch_size = proj_embedding_v.shape[0]
        for i in range(args.num_dfp):
            # 简单分配：每个DFP分配一部分特征
            start_idx = i * (batch_size // args.num_dfp)
            end_idx = (i + 1) * (batch_size // args.num_dfp) if i < args.num_dfp - 1 else batch_size
            if start_idx < batch_size:
                combined_features = torch.cat([
                    proj_embedding_v[start_idx:end_idx].view(-1, args.embedding_dim),
                    proj_embedding_a[start_idx:end_idx].view(-1, args.embedding_dim)
                ], dim=0)
                batch_features_by_dfp[i] = combined_features
        
        loss_compact, loss_separate = cov_dfp.compute_metric_learning_losses(
            batch_features_by_dfp, args.separation_margin
        )

    lambda_c = get_lambda_c(iter_num // 150)
    total_loss = (args.lamda * loss_s + lambda_c * loss_c + args.lambda_hcc * loss_hcc + 
                 args.lambda_compact * loss_compact + args.lambda_separate * loss_separate)

    # 反向传播
    optimizer.zero_grad()
    if selector_optimizer:
        selector_optimizer.zero_grad()
    
    total_loss.backward()
    optimizer.step()
    if selector_optimizer:
        selector_optimizer.step()

    # 记录日志
    if writer:
        writer.add_scalar('train/loss_s', loss_s, iter_num)
        writer.add_scalar('train/loss_c', loss_c, iter_num)
        writer.add_scalar('train/loss_hcc', loss_hcc, iter_num)
        writer.add_scalar('train/loss_compact', loss_compact, iter_num)
        writer.add_scalar('train/loss_separate', loss_separate, iter_num)
        writer.add_scalar('train/total_loss', total_loss, iter_num)
    
    if args.use_wandb:
        wandb.log({
            'train/loss_s': loss_s.item(),
            'train/loss_c': loss_c.item(),
            'train/loss_hcc': loss_hcc.item(),
            'train/loss_compact': loss_compact.item() if isinstance(loss_compact, torch.Tensor) else loss_compact,
            'train/loss_separate': loss_separate.item() if isinstance(loss_separate, torch.Tensor) else loss_separate,
            'train/total_loss': total_loss.item(),
            'iteration': iter_num
        })

    return total_loss.item()

if __name__ == "__main__":
    # 创建快照目录
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # 复制代码到快照目录
    shutil.copy(__file__, snapshot_path)

    # 设置日志
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # 初始化wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                  name=f"{args.exp}_{args.labelnum}_{args.model}", config=args)

    # 创建模型
    print(f"创建模型: {args.model}")
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    print(f"模型创建成功: {type(model).__name__}")
    
    # 只有当模型没有内置投影头时才添加（corn2d已经有内置投影头）
    if args.use_dfp and not hasattr(model, 'projection_head1'):
        # 获取模型的特征维度
        if args.model == 'unet':
            feature_dim = 64  # UNet的特征维度
        elif args.model == 'vnet':
            feature_dim = 64
        elif args.model == 'corn':
            feature_dim = 64
        elif args.model == 'corn2d':
            feature_dim = 16  # corn2d模型的特征维度（n_filters）
        else:
            feature_dim = 64
            
        model.projection_head1 = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, args.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_dim, args.embedding_dim)
        ).cuda()
        
        model.projection_head2 = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, args.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_dim, args.embedding_dim)
        ).cuda()
        
        print(f"添加了外部投影头，特征维度: {feature_dim}")
    else:
        print(f"使用模型内置的投影头")

    # 初始化DFP
    cov_dfp = None
    if args.use_dfp:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cov_dfp = CovarianceDynamicFeaturePool(
            feature_dim=args.embedding_dim,
            num_dfp=args.num_dfp,
            max_global_features=args.max_global_features,
            device=device
        )

    # 创建数据加载器
    db_train = ACDCDataSet(base_dir=train_data_path,
                          split='train',
                          num=args.max_samples,
                          transform=RandomGenerator(patch_size),
                          use_h5=args.use_h5,
                          with_idx=True)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # Selector单独的优化器
    selector_optimizer = None
    if args.use_dfp and hasattr(model, 'dfp_selector') and model.dfp_selector is not None:
        selector_params = list(model.dfp_selector.parameters())
        selector_optimizer = optim.Adam(selector_params, lr=0.001, weight_decay=0.0001)
        print(f"创建了Selector优化器，参数数量: {len(selector_params)}")
    else:
        print("未创建Selector优化器（模型无dfp_selector或未启用DFP）")

    # 初始化tensorboard writer
    writer = SummaryWriter(snapshot_path + '/log') if not args.use_wandb else None
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    # 创建ACDC数据处理器
    data_processor = ACDCDataProcessor(expected_classes=num_classes, verbose=True)
    print(f"创建了ACDC数据处理器，期望类别数: {num_classes}")
    
    consistency_criterion = losses.mse_loss
    
    # 创建简单的二元Dice损失函数
    def binary_dice_loss(pred, target, smooth=1e-8):
        """二元Dice损失函数，适用于单类别预测"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    dice_loss = binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    
    # 训练状态
    dfps_built = False
    selector_trained = False
    selector_train_counter = 0
    last_dfp_reconstruct_iter = 0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # 学习率调度
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # 选择训练阶段
            if not args.use_dfp or iter_num < args.dfp_start_iter:
                # 阶段一：基础训练
                loss = train_stage_one(model, sampled_batch, optimizer, consistency_criterion, 
                                     dice_loss, cov_dfp, iter_num, writer, data_processor)
            else:
                # 阶段二：DFP训练
                loss = train_stage_two(model, sampled_batch, optimizer, selector_optimizer,
                                     consistency_criterion, dice_loss, cov_dfp, iter_num, writer, data_processor)

            iter_num += 1
            
            # 日志记录
            if iter_num % 50 == 0:
                logging.info(f'iteration {iter_num}: loss: {loss:.4f}, lr: {lr_:.6f}')
                # 打印数据处理统计
                stats = data_processor.get_stats()
                if stats['shape_fixes'] > 0:
                    logging.info(f'数据处理统计: {stats}')

            # 保存模型
            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    # 保存最终模型
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save final model to {}".format(save_mode_path))
    
    # 打印最终统计
    final_stats = data_processor.get_stats()
    logging.info(f"训练完成，数据处理最终统计: {final_stats}")
    
    if writer:
        writer.close()
    
    if args.use_wandb:
        wandb.finish() 