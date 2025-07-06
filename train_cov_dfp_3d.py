#!/usr/bin/env python3

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
from myutils.contrastive_prototype_manager import ContrastivePrototypeManager
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
parser.add_argument('--exp', type=str, default='corn_dfp', help='exp_name')
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
parser.add_argument('--consistency_o', type=float, default=0.05, help='lambda_s to balance sim loss')

# DFP参数
parser.add_argument('--use_dfp', action='store_true', help='whether to use dynamic feature pool')
parser.add_argument('--num_eigenvectors', type=int, default=8, help='number of eigenvectors for second-order anchors')
parser.add_argument('--max_store', type=int, default=10000, help='max features to store per class')
parser.add_argument('--ema_alpha', type=float, default=0.9, help='EMA alpha for feature update')
parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--memory_num', type=int, default=256, help='num of embeddings per class in memory bank')
parser.add_argument('--num_filtered', type=int, default=12800, help='num of unlabeled embeddings to calculate similarity')

# 对比学习原型参数
parser.add_argument('--use_prototype', action='store_true', help='whether to use contrastive prototype separation')
parser.add_argument('--prototype_confidence_threshold', type=float, default=0.8, help='confidence threshold for prototype selection')
parser.add_argument('--prototype_elements_per_class', type=int, default=32, help='number of feature elements per class in memory')
parser.add_argument('--prototype_contrastive_weight', type=float, default=1.0, help='weight for contrastive learning loss')
parser.add_argument('--prototype_intra_weight', type=float, default=0.1, help='weight for intra-class loss')
parser.add_argument('--prototype_inter_weight', type=float, default=0.1, help='weight for inter-class loss')
parser.add_argument('--prototype_margin', type=float, default=1.0, help='margin for inter-class separation')
parser.add_argument('--prototype_use_learned_selector', action='store_true', help='whether to use learned feature selector')

# 其他参数
parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
parser.add_argument('--wandb_project', type=str, default='CORN-DFP', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')

args = parser.parse_args()

snapshot_path = "./model/LA_{}_{}_{}_memory{}_feat{}_labeled_numfiltered_{}_consistency_{}_rampup_{}_consis_o_{}_proto{}_iter_{}_seed_{}".format(
    args.exp,
    args.labelnum,
    args.model,
    args.memory_num,
    args.embedding_dim,
    args.num_filtered,
    args.consistency,
    args.consistency_rampup,
    args.consistency_o,
    "contrastive" if args.use_prototype else "none",
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

def extract_features_and_labels(embedding_v, embedding_a, label_batch, labeled_bs):
    """提取特征和标签用于DFP更新 - 使用已经计算好的特征"""
    with torch.no_grad():
        # 只使用标记样本
        labeled_features_v = embedding_v[:labeled_bs]  # [labeled_bs, feat_dim, H, W, D]
        labeled_features_a = embedding_a[:labeled_bs]  # [labeled_bs, feat_dim, H, W, D]
        labeled_labels = label_batch[:labeled_bs]      # [labeled_bs, H, W, D]
        
        # 转换为列表格式
        features_list = []
        labels_list = []
        
        for i in range(labeled_bs):
            # 获取单个样本的特征和标签
            feat_v = labeled_features_v[i].permute(1, 2, 3, 0).contiguous()  # [H, W, D, feat_dim]
            feat_a = labeled_features_a[i].permute(1, 2, 3, 0).contiguous()  # [H, W, D, feat_dim]
            label = labeled_labels[i]  # [H, W, D]
            
            # 展平为像素级特征
            feat_v_flat = feat_v.view(-1, feat_v.shape[-1])  # [H*W*D, feat_dim]
            feat_a_flat = feat_a.view(-1, feat_a.shape[-1])  # [H*W*D, feat_dim]
            label_flat = label.view(-1)  # [H*W*D]
            
            # 平均两个分支的特征
            feat_combined = (feat_v_flat + feat_a_flat) / 2
            
            features_list.append(feat_combined)
            labels_list.append(label_flat)
    
    return features_list, labels_list

def sample_anchors_from_dfp(dfp, num_eigenvectors=8):
    """从DFP中采样anchor特征（GPU版本）"""
    anchor_list = dfp.sample_labeled_features(num_eigenvectors)
    
    # 直接返回GPU张量列表
    anchor_tensors = []
    for anchors in anchor_list:
        if anchors is not None:
            anchor_tensors.append(anchors)  # 已经是GPU张量
        else:
            anchor_tensors.append(None)
    
    return anchor_tensors

def compute_anchor_loss(features_list, labels_list, anchor_tensors, device):
    """计算anchor相似性损失（GPU版本）- 修复版本"""
    total_loss = torch.tensor(0.0, device=device)
    num_valid = 0
    
    for feat_batch, label_batch in zip(features_list, labels_list):
        # 确保特征在正确的设备上
        feat_batch = feat_batch.to(device)
        label_batch = label_batch.to(device)
        
        for class_id in range(num_classes):
            # 获取当前类别的特征
            class_mask = (label_batch == class_id)
            if not class_mask.any():
                continue
                
            class_features = feat_batch[class_mask]  # [N_class, feat_dim]
            
            # 获取对应的anchor（已经在GPU上）
            if anchor_tensors[class_id] is not None:
                anchors = anchor_tensors[class_id].to(device)  # [num_anchors, feat_dim]
                
                # L2归一化特征和anchor
                class_features_norm = torch.nn.functional.normalize(class_features, p=2, dim=1)
                anchors_norm = torch.nn.functional.normalize(anchors, p=2, dim=1)
                
                # 计算余弦相似性（归一化后的内积）
                similarities = torch.mm(class_features_norm, anchors_norm.t())  # [N_class, num_anchors]
                
                # 使用最大相似性作为目标
                max_similarities = torch.max(similarities, dim=1)[0]  # [N_class]
                
                # 计算损失（鼓励高相似性）- 使用余弦距离
                # 余弦距离 = 1 - 余弦相似性，范围在[0, 2]
                cosine_distance = 1.0 - max_similarities
                loss = torch.mean(cosine_distance)
                
                total_loss = total_loss + loss
                num_valid += 1
    
    if num_valid > 0:
        return total_loss / num_valid
    else:
        return torch.tensor(0.0, device=device)

def train_with_dfp(model, sampled_batch, optimizer, consistency_criterion, dice_loss, 
                   dfp, iter_num, prototype_manager=None, selector_optimizer=None, writer=None):
    """使用DFP的训练函数"""
    model.train()
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()
    
    # 前向传播
    outputs_v, outputs_a, embedding_v, embedding_a = model(volume_batch)
    outputs_list = [outputs_v, outputs_a]
    num_outputs = len(outputs_list)

    # 确保张量在正确的设备上
    device = volume_batch.device
    y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)
    y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)

    # 监督损失
    loss_s = 0
    for i in range(num_outputs):
        y = outputs_list[i][:labeled_bs, ...]
        y_prob = F.softmax(y, dim=1)
        loss_s += dice_loss(y_prob[:, 1, ...], label_batch[:labeled_bs, ...] == 1)

        y_all = outputs_list[i]
        y_prob_all = F.softmax(y_all, dim=1)
        y_ori[i] = y_prob_all
        y_pseudo_label[i] = sharpening(y_prob_all)

    # 一致性损失
    loss_c = 0
    for i in range(num_outputs):
        for j in range(num_outputs):
            if i != j:
                loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

    # DFP相关损失
    loss_anchor = torch.tensor(0.0, device=device)
    if args.use_dfp and dfp is not None:
        # 提取特征用于DFP更新 - 使用已经计算好的特征
        features_list, labels_list = extract_features_and_labels(embedding_v, embedding_a, label_batch, labeled_bs)
        
        # 更新DFP
        dfp.update_labeled_features(features_list, labels_list)
        
        # 获取anchor特征
        anchor_tensors = sample_anchors_from_dfp(dfp, args.num_eigenvectors)
        
        # 计算anchor损失
        loss_anchor = compute_anchor_loss(features_list, labels_list, anchor_tensors, device)
    
    # 对比学习原型损失
    loss_prototype = torch.tensor(0.0, device=device)
    prototype_info = {}
    if args.use_prototype and prototype_manager is not None:
        # 重用已经计算的特征
        features_combined = (embedding_v + embedding_a) / 2  # [batch_size, feature_dim, H, W, D]
        outputs_combined = (outputs_v + outputs_a) / 2  # [batch_size, num_classes, H, W, D]
        
        # 标记数据的对比学习原型损失
        if labeled_bs > 0:
            labeled_features = features_combined[:labeled_bs]
            labeled_outputs = outputs_combined[:labeled_bs]
            labeled_labels = label_batch[:labeled_bs]
            
            loss_prototype_labeled, loss_dict_labeled = prototype_manager.update_and_compute_loss(
                labeled_features, labeled_outputs, labeled_labels, 
                is_labeled=True,
                contrastive_weight=args.prototype_contrastive_weight,
                intra_weight=args.prototype_intra_weight,
                inter_weight=args.prototype_inter_weight,
                margin=args.prototype_margin
            )
            loss_prototype += loss_prototype_labeled
            prototype_info.update({k + '_labeled': v for k, v in loss_dict_labeled.items()})
        
        # 无标记数据的对比学习原型损失
        if len(features_combined) > labeled_bs:
            unlabeled_features = features_combined[labeled_bs:]
            unlabeled_outputs = outputs_combined[labeled_bs:]
            
            loss_prototype_unlabeled, loss_dict_unlabeled = prototype_manager.update_and_compute_loss(
                unlabeled_features, unlabeled_outputs, 
                labels=None,
                is_labeled=False,
                contrastive_weight=args.prototype_contrastive_weight,
                intra_weight=args.prototype_intra_weight,
                inter_weight=args.prototype_inter_weight,
                margin=args.prototype_margin
            )
            loss_prototype += loss_prototype_unlabeled
            prototype_info.update({k + '_unlabeled': v for k, v in loss_dict_unlabeled.items()})

    # 计算总损失
    lambda_c = get_lambda_c(iter_num // 150)
    lambda_d = get_lambda_d(iter_num // 150)
    lambda_p = args.prototype_contrastive_weight if args.use_prototype else 0.0
    
    total_loss = args.lamda * loss_s + lambda_c * loss_c + lambda_d * loss_anchor + lambda_p * loss_prototype

    # 反向传播
    optimizer.zero_grad()
    if selector_optimizer is not None:
        selector_optimizer.zero_grad()
    
    total_loss.backward()
    
    optimizer.step()
    if selector_optimizer is not None:
        selector_optimizer.step()

    # 记录日志
    if args.use_prototype:
        logging.info('Iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_anchor: %03f, loss_prototype: %03f' % (
            iter_num, total_loss, loss_s, loss_c, loss_anchor, loss_prototype))
    else:
        logging.info('Iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_anchor: %03f' % (
        iter_num, total_loss, loss_s, loss_c, loss_anchor))
    
    result = {
        'total_loss': total_loss.item(),
        'loss_s': loss_s.item(),
        'loss_c': loss_c.item(),
        'loss_anchor': loss_anchor.item(),
        'lambda_c': lambda_c,
        'lambda_d': lambda_d
    }
    
    if args.use_prototype:
        result.update({
            'loss_prototype': loss_prototype.item(),
            'lambda_p': lambda_p
        })
        result.update(prototype_info)
    
    return result

def train_without_dfp(model, sampled_batch, optimizer, consistency_criterion, dice_loss, 
                      iter_num, prototype_manager=None, selector_optimizer=None, writer=None):
    """不使用DFP的标准训练函数"""
    model.train()
    volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
    volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()

    # 前向传播
    outputs_v, outputs_a, embedding_v, embedding_a = model(volume_batch)
    outputs_list = [outputs_v, outputs_a]
    num_outputs = len(outputs_list)

    # 确保张量在正确的设备上
    device = volume_batch.device
    y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)
    y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape, device=device)

    # 监督损失
    loss_s = 0
    for i in range(num_outputs):
        y = outputs_list[i][:labeled_bs, ...]
        y_prob = F.softmax(y, dim=1)
        loss_s += dice_loss(y_prob[:, 1, ...], label_batch[:labeled_bs, ...] == 1)

        y_all = outputs_list[i]
        y_prob_all = F.softmax(y_all, dim=1)
        y_ori[i] = y_prob_all
        y_pseudo_label[i] = sharpening(y_prob_all)

    # 一致性损失
    loss_c = 0
    for i in range(num_outputs):
        for j in range(num_outputs):
            if i != j:
                loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

    # 对比学习原型损失
    loss_prototype = torch.tensor(0.0, device=device)
    prototype_info = {}
    if args.use_prototype and prototype_manager is not None:
        # 重用已经计算的特征
        features_combined = (embedding_v + embedding_a) / 2  # [batch_size, feature_dim, H, W, D]
        outputs_combined = (outputs_v + outputs_a) / 2  # [batch_size, num_classes, H, W, D]
        
        # 标记数据的对比学习原型损失
        if labeled_bs > 0:
            labeled_features = features_combined[:labeled_bs]
            labeled_outputs = outputs_combined[:labeled_bs]
            labeled_labels = label_batch[:labeled_bs]
            
            loss_prototype_labeled, loss_dict_labeled = prototype_manager.update_and_compute_loss(
                labeled_features, labeled_outputs, labeled_labels, 
                is_labeled=True,
                contrastive_weight=args.prototype_contrastive_weight,
                intra_weight=args.prototype_intra_weight,
                inter_weight=args.prototype_inter_weight,
                margin=args.prototype_margin
            )
            loss_prototype += loss_prototype_labeled
            prototype_info.update({k + '_labeled': v for k, v in loss_dict_labeled.items()})
        
        # 无标记数据的对比学习原型损失
        if len(features_combined) > labeled_bs:
            unlabeled_features = features_combined[labeled_bs:]
            unlabeled_outputs = outputs_combined[labeled_bs:]
            
            loss_prototype_unlabeled, loss_dict_unlabeled = prototype_manager.update_and_compute_loss(
                unlabeled_features, unlabeled_outputs, 
                labels=None,
                is_labeled=False,
                contrastive_weight=args.prototype_contrastive_weight,
                intra_weight=args.prototype_intra_weight,
                inter_weight=args.prototype_inter_weight,
                margin=args.prototype_margin
            )
            loss_prototype += loss_prototype_unlabeled
            prototype_info.update({k + '_unlabeled': v for k, v in loss_dict_unlabeled.items()})

    # 计算总损失
    lambda_c = get_lambda_c(iter_num // 150)
    lambda_p = args.prototype_contrastive_weight if args.use_prototype else 0.0
    
    total_loss = args.lamda * loss_s + lambda_c * loss_c + lambda_p * loss_prototype

    # 反向传播
    optimizer.zero_grad()
    if selector_optimizer is not None:
        selector_optimizer.zero_grad()
    
    total_loss.backward()
    
    optimizer.step()
    if selector_optimizer is not None:
        selector_optimizer.step()

    # 记录日志
    if args.use_prototype:
        logging.info('Iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_prototype: %03f' % (
            iter_num, total_loss, loss_s, loss_c, loss_prototype))
    else:
        logging.info('Iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f' % (
        iter_num, total_loss, loss_s, loss_c))
    
    result = {
        'total_loss': total_loss.item(),
        'loss_s': loss_s.item(),
        'loss_c': loss_c.item(),
        'lambda_c': lambda_c
    }
    
    if args.use_prototype:
        result.update({
            'loss_prototype': loss_prototype.item(),
            'lambda_p': lambda_p
        })
        result.update(prototype_info)
    
    return result

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
        proto_tag = "contrastive" if args.use_prototype else "none"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp}_{args.model}_feat{args.embedding_dim}_dfp{args.use_dfp}_proto{proto_tag}",
            tags=[args.dataset_name, "corn_dfp", f"feat_{args.embedding_dim}", f"proto_{proto_tag}"]
        )
        logging.info(f"Wandb initialized with project: {args.wandb_project}")

    # 创建模型
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", 
                       feat_dim=args.embedding_dim)
    
    # 移动模型到GPU
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("Model moved to GPU")
    else:
        logging.warning("CUDA not available, using CPU")

    # 创建动态特征池
    dfp = None
    if args.use_dfp:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dfp = DynamicFeaturePool(
            num_labeled_samples=args.labelnum,
            num_cls=num_classes,
            max_store=args.max_store,
            ema_alpha=args.ema_alpha,
            device=device
        )
        logging.info(f"DynamicFeaturePool created on {device} with max_store={args.max_store}, ema_alpha={args.ema_alpha}")
    
    # 创建对比学习原型管理器
    prototype_manager = None
    if args.use_prototype:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prototype_manager = ContrastivePrototypeManager(
            num_classes=num_classes,
            feature_dim=args.embedding_dim,
            elements_per_class=args.prototype_elements_per_class,
            confidence_threshold=args.prototype_confidence_threshold,
            use_learned_selector=args.prototype_use_learned_selector,
            device=device
        )
        logging.info(f"ContrastivePrototypeManager created on {device} with elements_per_class={args.prototype_elements_per_class}, confidence_threshold={args.prototype_confidence_threshold}")
        logging.info(f"Prototype weights: contrastive={args.prototype_contrastive_weight}, intra={args.prototype_intra_weight}, inter={args.prototype_inter_weight}, margin={args.prototype_margin}")
        logging.info(f"Use learned selector: {args.prototype_use_learned_selector}")

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
    
    # 如果使用学习的选择器，为其创建额外的优化器
    selector_optimizer = None
    if args.use_prototype and args.prototype_use_learned_selector and prototype_manager is not None:
        feature_selectors, memory_selectors = prototype_manager.get_selectors()
        selector_params = []
        if feature_selectors is not None:
            selector_params.extend(feature_selectors.parameters())
        if memory_selectors is not None:
            selector_params.extend(memory_selectors.parameters())
        
        if len(selector_params) > 0:
            selector_optimizer = optim.Adam(selector_params, lr=base_lr * 0.1)
            logging.info(f"Created selector optimizer with {len(selector_params)} parameter groups")

    # 初始化tensorboard writer
    writer = SummaryWriter(snapshot_path + '/log') if not args.use_wandb else None
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            iter_num += 1
            
            # 学习率调整
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                
                # 调整选择器优化器的学习率
                if selector_optimizer is not None:
                    selector_lr = lr_ * 0.1  # 选择器使用更小的学习率
                    for param_group in selector_optimizer.param_groups:
                        param_group['lr'] = selector_lr
            
            # 训练
            if args.use_dfp:
                metrics = train_with_dfp(model, sampled_batch, optimizer, consistency_criterion, 
                                       dice_loss, dfp, iter_num, prototype_manager, selector_optimizer, writer)
            else:
                metrics = train_without_dfp(model, sampled_batch, optimizer, consistency_criterion, 
                                          dice_loss, iter_num, prototype_manager, selector_optimizer, writer)
            
            # 记录指标
            if args.use_wandb:
                log_dict = {
                    'train/loss': metrics['total_loss'],
                    'train/loss_supervised': metrics['loss_s'],
                    'train/loss_consistency': metrics['loss_c'],
                    'train/lambda_c': metrics['lambda_c'],
                    'train/lr': lr_,
                    'iteration': iter_num
                }
                
                if args.use_dfp:
                    log_dict.update({
                        'train/loss_anchor': metrics['loss_anchor'],
                        'train/lambda_d': metrics['lambda_d']
                    })
                
                if args.use_prototype:
                    log_dict.update({
                        'train/loss_prototype': metrics['loss_prototype'],
                        'train/lambda_p': metrics['lambda_p']
                    })
                    # 记录原型相关的详细信息
                    for key, value in metrics.items():
                        if 'labeled' in key or 'unlabeled' in key:
                            log_dict[f'train/{key}'] = value
                
                wandb.log(log_dict)
            else:
                writer.add_scalar('train/loss', metrics['total_loss'], iter_num)
                writer.add_scalar('train/loss_supervised', metrics['loss_s'], iter_num)
                writer.add_scalar('train/loss_consistency', metrics['loss_c'], iter_num)
                writer.add_scalar('train/lambda_c', metrics['lambda_c'], iter_num)
                writer.add_scalar('train/lr', lr_, iter_num)
                
                if args.use_dfp:
                    writer.add_scalar('train/loss_anchor', metrics['loss_anchor'], iter_num)
                    writer.add_scalar('train/lambda_d', metrics['lambda_d'], iter_num)
                
                if args.use_prototype:
                    writer.add_scalar('train/loss_prototype', metrics['loss_prototype'], iter_num)
                    writer.add_scalar('train/lambda_p', metrics['lambda_p'], iter_num)
                    # 记录原型相关的详细信息
                    for key, value in metrics.items():
                        if 'labeled' in key or 'unlabeled' in key:
                            writer.add_scalar(f'train/{key}', value, iter_num)

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
                    writer.add_scalar('val/dice', dice_sample, iter_num)
                    writer.add_scalar('val/best_dice', best_dice, iter_num)
                
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