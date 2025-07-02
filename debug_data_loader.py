#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.acdc_dataset import ACDCDataSet, RandomGenerator, TwoStreamBatchSampler
from torch.utils.data import DataLoader

def debug_data_loader():
    """调试ACDC数据加载器，检查数据格式"""
    print("调试ACDC数据加载器...")
    
    # 配置参数
    train_data_path = '/home/jovyan/work/medical_dataset/ACDC_processed'
    patch_size = (256, 256)
    max_samples = 200
    labeled_bs = 2
    batch_size = 4
    
    # 创建数据集
    try:
        db_train = ACDCDataSet(base_dir=train_data_path,
                              split='train',
                              num=max_samples,
                              transform=RandomGenerator(patch_size),
                              use_h5=True,
                              with_idx=True)
        print(f"✓ 数据集创建成功，共 {len(db_train)} 个样本")
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        return
    
    # 创建批采样器
    labelnum = 7
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    
    # 创建数据加载器
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=False)
    
    print(f"✓ 数据加载器创建成功，每epoch {len(trainloader)} 个批次")
    
    # 检查前几个批次
    for i, sampled_batch in enumerate(trainloader):
        if i >= 3:  # 只检查前3个批次
            break
            
        volume_batch = sampled_batch['image']
        label_batch = sampled_batch['label']
        idx_batch = sampled_batch['idx']
        
        print(f"\n=== 批次 {i+1} ===")
        print(f"Image batch shape: {volume_batch.shape}")
        print(f"Image dtype: {volume_batch.dtype}")
        print(f"Image min/max: {volume_batch.min():.4f}/{volume_batch.max():.4f}")
        
        print(f"Label batch shape: {label_batch.shape}")
        print(f"Label dtype: {label_batch.dtype}")
        print(f"Label unique values: {torch.unique(label_batch)}")
        
        print(f"Index batch: {idx_batch}")
        
        # 检查每个样本
        for j in range(volume_batch.shape[0]):
            img = volume_batch[j]
            lbl = label_batch[j]
            print(f"  样本 {j}: image {img.shape}, label {lbl.shape}")
            
            # 检查是否有异常值
            if len(img.shape) != 3:  # 应该是 [C, H, W]
                print(f"    ⚠️  图像维度异常: {img.shape}")
            if img.shape[0] != 1:  # 应该是1个通道
                print(f"    ⚠️  图像通道数异常: {img.shape[0]} (期望1)")
            if len(lbl.shape) != 2:  # 应该是 [H, W]
                print(f"    ⚠️  标签维度异常: {lbl.shape}")
    
    print("\n=== 单个样本测试 ===")
    # 测试单个样本
    try:
        sample = db_train[0]
        print(f"单个样本 - Image: {sample['image'].shape}, Label: {sample['label'].shape}")
        print(f"Image dtype: {sample['image'].dtype}, Label dtype: {sample['label'].dtype}")
        print(f"Image min/max: {sample['image'].min():.4f}/{sample['image'].max():.4f}")
        print(f"Label unique: {torch.unique(sample['label'])}")
    except Exception as e:
        print(f"✗ 单个样本测试失败: {e}")
    
    print("\n✓ 数据加载器调试完成")

if __name__ == "__main__":
    debug_data_loader() 