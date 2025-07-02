#!/usr/bin/env python3

from dataloaders.acdc_dataset import ACDCDataSet, RandomGenerator
import torch

def test_dataset_loading():
    """测试数据集加载"""
    print("=== 测试数据集加载 ===")
    
    try:
        # 不使用任何transform
        dataset_no_transform = ACDCDataSet('/home/jovyan/work/medical_dataset/ACDC_processed', None, 'train', transform=None)
        print(f'数据集大小: {len(dataset_no_transform)}')
        
        # 测试获取第一个样本（无transform）
        sample = dataset_no_transform[0]
        print(f'无transform样本键: {sample.keys()}')
        print(f'图像形状: {sample["image"].shape}')
        print(f'标签形状: {sample["label"].shape}')
        print(f'patient_id: {sample["patient_id"]}')
        print(f'frame_type: {sample["frame_type"]}')
        print('无transform测试成功!')
        
    except Exception as e:
        print(f'无transform测试错误: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== 测试带Transform的数据集加载 ===")
    
    try:
        # 使用transform
        dataset_with_transform = ACDCDataSet('/home/jovyan/work/medical_dataset/ACDC_processed', None, 'train', transform=RandomGenerator((256, 256)))
        print(f'数据集大小: {len(dataset_with_transform)}')
        
        # 测试获取第一个样本（有transform）
        sample = dataset_with_transform[0]
        print(f'有transform样本键: {sample.keys()}')
        print(f'图像形状: {sample["image"].shape}')
        print(f'标签形状: {sample["label"].shape}')
        print(f'patient_id: {sample["patient_id"]}')
        print(f'frame_type: {sample["frame_type"]}')
        print('有transform测试成功!')
        
    except Exception as e:
        print(f'有transform测试错误: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_transform_directly():
    """直接测试transform"""
    print("\n=== 直接测试Transform ===")
    
    try:
        import numpy as np
        
        # 创建模拟数据
        image = np.random.rand(1, 256, 256).astype(np.float32)
        label = np.random.randint(0, 4, (256, 256)).astype(np.uint8)
        
        sample = {
            'image': image,
            'label': label,
            'idx': 0,
            'patient_id': 'test_patient',
            'frame_type': 'ED',
            'slice_idx': 5
        }
        
        print(f'原始sample键: {sample.keys()}')
        
        # 应用transform
        transform = RandomGenerator((256, 256))
        transformed_sample = transform(sample)
        
        print(f'transform后sample键: {transformed_sample.keys()}')
        print('直接transform测试成功!')
        
    except Exception as e:
        print(f'直接transform测试错误: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success1 = test_dataset_loading()
    success2 = test_transform_directly()
    
    if success1 and success2:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试失败!") 