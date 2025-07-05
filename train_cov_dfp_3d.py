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

snapshot_path = "./model/LA_{}_{}_labeled_{}_consistency_{}_rampup_{}_iter_{}_seed_{}".format(
    args.exp,
    args.labelnum,
    args.labelnum,
    args.consistency,
    args.consistency_rampup,
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
            
            # 处理有标签部分的特征和标签
            labeled_features = embedding[:args.labeled_bs]  # (B, C, H, W, D)
            labeled_labels = label_batch[:args.labeled_bs]  # (B, 1, H, W, D)
            
            # 将特征转换为二维张量 (N, C)
            B, C, H, W, D = labeled_features.shape
            labeled_features = labeled_features.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            # 将标签转换为一维向量 (N,)，确保去除channel维
            labeled_labels = labeled_labels.squeeze(1).reshape(-1)
            
            # 确保在同一设备上
            labeled_features = labeled_features.to(embedding.device)
            labeled_labels = labeled_labels.to(embedding.device)
            
            # 更新特征池
            feature_pool.add_to_global_pool(labeled_features, labeled_labels)
            
            # 计算分割损失
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