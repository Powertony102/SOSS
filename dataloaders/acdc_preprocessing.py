#!/usr/bin/env python3
"""
ACDC数据集预处理脚本
将原始NIFTI文件转换为H5格式，便于训练时快速加载
"""

import os
import glob
import h5py
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse

def process_acdc_data(data_root, output_dir, split='training'):
    """
    处理ACDC数据集
    
    Args:
        data_root: ACDC数据集根目录，包含training和testing文件夹
        output_dir: 输出目录
        split: 'training' 或 'testing'
    """
    print(f"处理 {split} 数据...")
    
    # 创建输出目录
    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    
    # 获取所有患者目录
    patient_dirs = sorted(glob.glob(os.path.join(data_root, split, "patient*")))
    print(f"找到 {len(patient_dirs)} 个患者")
    
    slice_count = 0
    processed_files = []
    
    for patient_dir in tqdm(patient_dirs, desc=f"处理{split}数据"):
        patient_id = os.path.basename(patient_dir)
        
        # 获取该患者的所有帧文件（排除4d文件）
        frame_files = glob.glob(os.path.join(patient_dir, f"{patient_id}_frame*.nii.gz"))
        frame_files = [f for f in frame_files if "4d" not in f and "gt" not in f]
        
        for frame_file in frame_files:
            try:
                # 读取图像
                img_itk = sitk.ReadImage(frame_file)
                image = sitk.GetArrayFromImage(img_itk)
                
                # 读取对应的标签
                frame_name = os.path.basename(frame_file).replace('.nii.gz', '')
                label_file = os.path.join(patient_dir, f"{frame_name}_gt.nii.gz")
                
                if os.path.exists(label_file):
                    lbl_itk = sitk.ReadImage(label_file)
                    label = sitk.GetArrayFromImage(lbl_itk)
                else:
                    print(f"警告：找不到标签文件 {label_file}")
                    continue
                
                # 检查维度匹配
                if image.shape != label.shape:
                    print(f"警告：图像和标签维度不匹配 {frame_file}")
                    continue
                
                # 归一化图像
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                image = image.astype(np.float32)
                
                # 将每个切片保存为单独的H5文件
                for slice_idx in range(image.shape[0]):
                    slice_image = image[slice_idx]
                    slice_label = label[slice_idx]
                    
                    # 跳过空白切片（标签全为0的切片）
                    if np.sum(slice_label) == 0:
                        continue
                    
                    # 保存切片
                    output_filename = f"{frame_name}_slice_{slice_idx:02d}.h5"
                    output_path = os.path.join(slices_dir, output_filename)
                    
                    with h5py.File(output_path, 'w') as h5f:
                        h5f.create_dataset('image', data=slice_image, compression="gzip")
                        h5f.create_dataset('label', data=slice_label, compression="gzip")
                    
                    processed_files.append(output_filename.replace('.h5', ''))
                    slice_count += 1
                    
            except Exception as e:
                print(f"处理文件 {frame_file} 时出错: {str(e)}")
                continue
    
    print(f"处理完成，总共生成 {slice_count} 个切片")
    
    # 生成训练列表文件
    if split == 'training':
        # 划分训练集和验证集（80%训练，20%验证）
        np.random.seed(42)  # 固定随机种子
        np.random.shuffle(processed_files)
        
        split_idx = int(0.8 * len(processed_files))
        train_files = processed_files[:split_idx]
        val_files = processed_files[split_idx:]
        
        # 保存训练集列表
        with open(os.path.join(output_dir, 'train_slices.list'), 'w') as f:
            for filename in train_files:
                f.write(f"{filename}\n")
        
        # 保存验证集列表
        with open(os.path.join(output_dir, 'val.list'), 'w') as f:
            for filename in val_files:
                f.write(f"{filename}\n")
        
        print(f"训练集：{len(train_files)} 个切片")
        print(f"验证集：{len(val_files)} 个切片")
    
    return slice_count


def create_volume_h5_files(data_root, output_dir, split='training'):
    """
    创建完整体积的H5文件（用于测试）
    """
    print(f"创建 {split} 体积文件...")
    
    volumes_dir = os.path.join(output_dir, "volumes")
    os.makedirs(volumes_dir, exist_ok=True)
    
    patient_dirs = sorted(glob.glob(os.path.join(data_root, split, "patient*")))
    
    for patient_dir in tqdm(patient_dirs, desc=f"处理{split}体积"):
        patient_id = os.path.basename(patient_dir)
        
        # 获取该患者的所有帧文件
        frame_files = glob.glob(os.path.join(patient_dir, f"{patient_id}_frame*.nii.gz"))
        frame_files = [f for f in frame_files if "4d" not in f and "gt" not in f]
        
        for frame_file in frame_files:
            try:
                # 读取图像
                img_itk = sitk.ReadImage(frame_file)
                image = sitk.GetArrayFromImage(img_itk)
                
                # 读取标签
                frame_name = os.path.basename(frame_file).replace('.nii.gz', '')
                label_file = os.path.join(patient_dir, f"{frame_name}_gt.nii.gz")
                
                if os.path.exists(label_file):
                    lbl_itk = sitk.ReadImage(label_file)
                    label = sitk.GetArrayFromImage(lbl_itk)
                else:
                    label = np.zeros_like(image)
                
                # 归一化
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                image = image.astype(np.float32)
                
                # 保存完整体积
                output_path = os.path.join(volumes_dir, f"{frame_name}.h5")
                with h5py.File(output_path, 'w') as h5f:
                    h5f.create_dataset('image', data=image, compression="gzip")
                    h5f.create_dataset('label', data=label, compression="gzip")
                    
            except Exception as e:
                print(f"处理体积文件 {frame_file} 时出错: {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(description='ACDC数据预处理')
    parser.add_argument('--data_root', type=str, 
                       default='/home/jovyan/work/medical_dataset/ACDC',
                       help='ACDC数据集根目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/jovyan/work/medical_dataset/ACDC_processed',
                       help='输出目录')
    parser.add_argument('--create_volumes', action='store_true',
                       help='是否创建体积H5文件')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理训练数据
    if os.path.exists(os.path.join(args.data_root, 'training')):
        process_acdc_data(args.data_root, args.output_dir, 'training')
        
        if args.create_volumes:
            create_volume_h5_files(args.data_root, args.output_dir, 'training')
    
    # 处理测试数据
    if os.path.exists(os.path.join(args.data_root, 'testing')):
        process_acdc_data(args.data_root, args.output_dir, 'testing')
        
        if args.create_volumes:
            create_volume_h5_files(args.data_root, args.output_dir, 'testing')
    
    print("ACDC数据预处理完成！")
    print(f"处理后的数据保存在: {args.output_dir}")
    print("数据结构:")
    print("├── slices/          # 2D切片H5文件")
    print("├── volumes/         # 3D体积H5文件（可选）")
    print("├── train_slices.list # 训练切片列表")
    print("└── val.list         # 验证切片列表")


if __name__ == "__main__":
    main() 