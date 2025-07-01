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
from myutils.cov_dynamic_feature_pool import CovarianceDynamicFeaturePool
from myutils.hcc_loss import hierarchical_coral, parse_hcc_weights
from dataloaders.dataset import *
from networks.net_factory import net_factory

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
parser.add_argument('--lambda_hcc', type=float, default=0.1, help='weight for HCC loss')

# HCC参数
parser.add_argument('--hcc_weights', type=str, default='0.5,0.5,1,1,1.5', help='HCC layer weights')
parser.add_argument('--cov_mode', type=str, default='patch', choices=['full', 'patch'], help='covariance mode')
parser.add_argument('--patch_size', type=int, default=4, help='patch size for covariance')

# DFP参数
parser.add_argument('--num_dfp', type=int, default=8, help='number of dynamic feature pools')
parser.add_argument('--dfp_start_iter', type=int, default=2000, help='iteration to start building DFPs')
parser.add_argument('--selector_train_iter', type=int, default=50, help='iterations for training selector')
parser.add_argument('--dfp_reconstruct_interval', type=int, default=1000, help='interval for reconstructing DFPs')
parser.add_argument('--max_global_features', type=int, default=50000, help='maximum global features')
parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dimension')

# 其他参数
parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
parser.add_argument('--wandb_project', type=str, default='Cov-DFP', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
parser.add_argument('--consistency_o', type=float, default=0.05, help='lambda_s to balance sim loss')
parser.add_argument('--hcc_patch_strategy', type=str, default='mean_cov', 
                    choices=['mean_cov', 'mean_loss', 'max_loss', 'topk'],
                    help='patch strategy for HCC loss: mean_cov|mean_loss|max_loss|topk')
parser.add_argument('--hcc_topk', type=int, default=5, help='top-k value when hcc_patch_strategy is topk')
parser.add_argument('--hcc_metric', type=str, default='fro', choices=['fro', 'log'],
                    help='metric for HCC loss computation: fro (Frobenius) or log (Log-Euclidean)')
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

snapshot_path = "./model/LA_{}_{}_dfp{}_memory{}_feat{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_iter_{}_seed_{}/{}".format(
    args.exp,
    args.labelnum,
    args.num_dfp,
    args.memory_num,
    args.embedding_dim,
    args.num_filtered,
    args.consistency,
    args.consistency_rampup,
    args.consistency_o,
    args.max_iteration,
    args.seed,
    args.model)

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
                   cov_dfp, iter_num, writer=None):
    """阶段一：初始预训练
    
    目标：训练初步的主模型，建立全局特征池
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
    loss_hcc = hierarchical_coral(features_v, features_a, hcc_weights,
                                cov_mode=args.cov_mode, 
                                patch_size=args.patch_size,
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
        
        # 添加到全局特征池
        cov_dfp.add_to_global_pool(combined_features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 记录日志
    logging.info('Stage 1 - Iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_hcc: %03f' % (
        iter_num, total_loss, loss_s, loss_c, loss_hcc))
    
    return {
        'total_loss': total_loss.item(),
        'loss_s': loss_s.item(),
        'loss_c': loss_c.item(),
        'loss_hcc': loss_hcc.item(),
        'lambda_c': lambda_c
    }

def train_stage_two_build_dfp(cov_dfp):
    """阶段二：构建DFP并生成Selector训练目标"""
    success = cov_dfp.build_dfps()
    if success:
        stats = cov_dfp.get_statistics()
        logging.info(f"DFP construction successful: {stats}")
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
                          cov_dfp, iter_num, writer=None):
    """阶段三B：主模型训练（使用Selector选择的DFP）"""
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
    loss_hcc = hierarchical_coral(features_v, features_a, hcc_weights,
                                cov_mode=args.cov_mode, 
                                patch_size=args.patch_size,
                                patch_strategy=args.hcc_patch_strategy,
                                topk=args.hcc_topk,
                                metric=args.hcc_metric,
                                scale=args.hcc_scale)

    lambda_c = get_lambda_c(iter_num // 150)
    total_loss = args.lamda * loss_s + lambda_c * loss_c + args.lambda_hcc * loss_hcc

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    logging.info('Stage 3B - Main training iter %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_hcc: %03f' % (
        iter_num, total_loss, loss_s, loss_c, loss_hcc))
    
    return {
        'total_loss': total_loss.item(),
        'loss_s': loss_s.item(),
        'loss_c': loss_c.item(),
        'loss_hcc': loss_hcc.item(),
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
            name=f"{args.exp}_{args.model}_dfp{args.num_dfp}_feat{args.embedding_dim}",
            tags=[args.dataset_name, "cov_dfp", f"dfp_{args.num_dfp}"]
        )
        logging.info(f"Wandb initialized with project: {args.wandb_project}")

    # 创建模型
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", 
                       feat_dim=args.embedding_dim, num_dfp=args.num_dfp, use_selector=True)
    
    # 移动模型到GPU
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("Model moved to GPU")
    else:
        logging.warning("CUDA not available, using CPU")

    # 创建协方差DFP (指定GPU设备)
    cov_dfp = None
    if args.use_dfp:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cov_dfp = CovarianceDynamicFeaturePool(
            feature_dim=args.embedding_dim,
            num_dfp=args.num_dfp,
            max_global_features=args.max_global_features,
            device=device
        )
        logging.info(f"CovarianceDynamicFeaturePool created on device: {device}")

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

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # Selector单独的优化器
    selector_optimizer = None
    if args.use_dfp:
        selector_params = list(model.dfp_selector.parameters())
        selector_optimizer = optim.Adam(selector_params, lr=0.001, weight_decay=0.0001)

    # 初始化tensorboard writer
    writer = SummaryWriter(snapshot_path + '/log') if not args.use_wandb else None
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    
    # 训练状态
    dfps_built = False
    selector_train_counter = 0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            iter_num += 1
            
            # 阶段判断
            if iter_num < args.dfp_start_iter:
                # 阶段一：初始预训练
                metrics = train_stage_one(model, sampled_batch, optimizer, consistency_criterion, 
                                        dice_loss, cov_dfp, iter_num, writer)
                
                # 记录指标
                if args.use_wandb:
                    wandb.log({
                        'stage': 1,
                        'train/loss': metrics['total_loss'],
                        'train/loss_supervised': metrics['loss_s'],
                        'train/loss_consistency': metrics['loss_c'],
                        'train/loss_hcc': metrics['loss_hcc'],
                        'train/lambda_c': metrics['lambda_c'],
                        'iteration': iter_num
                    })
                    
                    if args.use_dfp and cov_dfp is not None:
                        stats = cov_dfp.get_statistics()
                        wandb.log({
                            'dfp/global_pool_size': stats['global_pool_size'],
                            'iteration': iter_num
                        })
            
            elif iter_num == args.dfp_start_iter and args.use_dfp:
                # 阶段二：构建DFP
                logging.info("Starting Stage 2: Building DFPs...")
                dfps_built = train_stage_two_build_dfp(cov_dfp)
                
                if dfps_built:
                    stats = cov_dfp.get_statistics()
                    if args.use_wandb:
                        wandb.log({
                            'stage': 2,
                            'dfp/dfps_built': True,
                            'dfp/min_dfp_size': stats['min_dfp_size'],
                            'dfp/max_dfp_size': stats['max_dfp_size'],
                            'dfp/mean_dfp_size': stats['mean_dfp_size'],
                            'iteration': iter_num
                        })
                
                # 在构建DFP后，立即进行一次主模型训练
                if dfps_built:
                    metrics = train_stage_three_main(model, sampled_batch, optimizer, consistency_criterion, 
                                                   dice_loss, cov_dfp, iter_num, writer)
                    if args.use_wandb:
                        wandb.log({
                            'stage': 3,
                            'train/loss': metrics['total_loss'],
                            'train/loss_supervised': metrics['loss_s'],
                            'train/loss_consistency': metrics['loss_c'],
                            'train/loss_hcc': metrics['loss_hcc'],
                            'iteration': iter_num
                        })
            
            elif iter_num > args.dfp_start_iter and args.use_dfp and dfps_built:
                # 阶段三：交替训练
                if selector_train_counter < args.selector_train_iter:
                    # 训练Selector
                    selector_metrics = train_stage_three_selector(model, sampled_batch, selector_optimizer, 
                                                                cov_dfp, iter_num)
                    selector_train_counter += 1
                    
                    if args.use_wandb:
                        wandb.log({
                            'stage': 3,
                            'submode': 'selector',
                            'selector/loss': selector_metrics['selector_loss'],
                            'selector/accuracy': selector_metrics['selector_accuracy'],
                            'iteration': iter_num
                        })
                else:
                    # 训练主模型
                    metrics = train_stage_three_main(model, sampled_batch, optimizer, consistency_criterion, 
                                                   dice_loss, cov_dfp, iter_num, writer)
                    selector_train_counter = 0  # 重置计数器
                    
                    if args.use_wandb:
                        wandb.log({
                            'stage': 3,
                            'submode': 'main',
                            'train/loss': metrics['total_loss'],
                            'train/loss_supervised': metrics['loss_s'],
                            'train/loss_consistency': metrics['loss_c'],
                            'train/loss_hcc': metrics['loss_hcc'],
                            'iteration': iter_num
                        })
                
                # 周期性重构DFP
                if iter_num % args.dfp_reconstruct_interval == 0:
                    logging.info("Reconstructing DFPs...")
                    cov_dfp.reconstruct_dfps()
                    
            elif not args.use_dfp:
                # 不使用DFP时的标准训练
                metrics = train_stage_one(model, sampled_batch, optimizer, consistency_criterion, 
                                        dice_loss, None, iter_num, writer)
                
                if args.use_wandb:
                    wandb.log({
                        'train/loss': metrics['total_loss'],
                        'train/loss_supervised': metrics['loss_s'],
                        'train/loss_consistency': metrics['loss_c'],
                        'train/loss_hcc': metrics['loss_hcc'],
                        'iteration': iter_num
                    })

            # 验证和保存模型
            if iter_num >= 1000 and iter_num % 500 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA', 
                                                          dataset_path=args.dataset_path)
                
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                
                # 记录验证指标
                if args.use_wandb:
                    wandb.log({
                        'val/dice': dice_sample,
                        'val/best_dice': best_dice,
                        'iteration': iter_num
                    })
                else:
                    writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                    writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
                
        if iter_num >= max_iterations:
            iterator.close()
            break
    
    # 关闭日志
    if args.use_wandb:
        wandb.finish()
    else:
        writer.close() 