#!/usr/bin/env python3
"""
测试修复后的数据加载器
"""

import torch
from torch.utils.data import DataLoader
from dataloaders.acdc_dataset import ACDCDataSet, ToTensor, RandomGenerator

def test_dataloader():
    """测试数据加载器修复"""
    print("测试数据加载器修复...")
    
    # 测试用ToTensor的验证数据加载器
    try:
        db_val = ACDCDataSet(
            base_dir='/home/jovyan/work/medical_dataset/ACDC_processed',
            list_dir=None,
            split='val',
            transform=ToTensor()
        )
        
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"验证数据集大小: {len(db_val)}")
        
        # 测试加载第一个样本
        sample = next(iter(valloader))
        print(f"样本加载成功!")
        print(f"图像形状: {sample['image'].shape}")
        print(f"标签形状: {sample['label'].shape}")
        print(f"图像类型: {type(sample['image'])}")
        print(f"标签类型: {type(sample['label'])}")
        
        print("✅ ToTensor数据加载器测试通过!")
        
    except Exception as e:
        print(f"❌ ToTensor数据加载器测试失败: {e}")
        return False
    
    # 测试用RandomGenerator的训练数据加载器
    try:
        db_train = ACDCDataSet(
            base_dir='/home/jovyan/work/medical_dataset/ACDC_processed',
            list_dir=None,
            split='train',
            transform=RandomGenerator((256, 256))
        )
        
        trainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"训练数据集大小: {len(db_train)}")
        
        # 测试加载第一个样本
        sample = next(iter(trainloader))
        print(f"训练样本加载成功!")
        print(f"图像形状: {sample['image'].shape}")
        print(f"标签形状: {sample['label'].shape}")
        print(f"图像类型: {type(sample['image'])}")
        print(f"标签类型: {type(sample['label'])}")
        
        print("✅ RandomGenerator数据加载器测试通过!")
        
    except Exception as e:
        print(f"❌ RandomGenerator数据加载器测试失败: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = test_dataloader()
    if success:
        print("\n🎉 所有数据加载器测试通过! 可以继续训练了。")
    else:
        print("\n💥 数据加载器仍有问题，需要进一步修复。") 