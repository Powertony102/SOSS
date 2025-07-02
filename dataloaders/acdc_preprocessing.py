#!/usr/bin/env python3
"""
ACDC数据集预处理脚本 - 优化版本
确保输出数据格式完全正确，避免运行时修复需求
"""

import os
import h5py
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import argparse


def validate_nifti_data(img_data, label_data):
    """验证NIFTI数据的有效性"""
    if img_data is None or label_data is None:
        return False, "数据为空"
    
    if img_data.shape != label_data.shape:
        return False, f"图像和标签形状不匹配: {img_data.shape} vs {label_data.shape}"
    
    # 检查是否为4D数据（时间序列）
    if len(img_data.shape) == 4:
        # 选择ED和ES帧（通常是第0帧和中间某帧）
        if img_data.shape[-1] >= 2:
            return True, "4D数据，将提取ED/ES帧"
        else:
            return False, "4D数据但时间点不足"
    elif len(img_data.shape) == 3:
        return True, "3D数据"
    else:
        return False, f"不支持的数据维度: {img_data.shape}"


def extract_ed_es_frames(img_4d, label_4d):
    """从4D数据中提取ED和ES帧"""
    # ED帧通常是第0帧（舒张末期）
    ed_img = img_4d[..., 0]
    ed_label = label_4d[..., 0]
    
    # ES帧通常在中间位置（收缩末期）
    # 简单策略：选择中间帧或根据心脏容积最小的帧
    num_frames = img_4d.shape[-1]
    es_frame_idx = num_frames // 2
    
    es_img = img_4d[..., es_frame_idx]
    es_label = label_4d[..., es_frame_idx]
    
    return (ed_img, ed_label), (es_img, es_label)


def normalize_image(image):
    """图像归一化"""
    # 计算非零区域的统计量
    nonzero_mask = image > 0
    if np.sum(nonzero_mask) == 0:
        return image.astype(np.float32)
    
    mean_val = np.mean(image[nonzero_mask])
    std_val = np.std(image[nonzero_mask])
    
    if std_val > 0:
        image = (image - mean_val) / std_val
    else:
        image = image - mean_val
    
    return image.astype(np.float32)


def extract_2d_slices(img_3d, label_3d, min_label_pixels=50):
    """从3D数据中提取2D切片，只保留有标签的切片"""
    slices_data = []
    
    # 遍历所有切片（通常在第2维，即axial方向）
    for slice_idx in range(img_3d.shape[2]):
        img_slice = img_3d[:, :, slice_idx]
        label_slice = label_3d[:, :, slice_idx]
        
        # 检查这个切片是否包含足够的标签
        label_pixels = np.sum(label_slice > 0)
        if label_pixels >= min_label_pixels:
            # 确保图像是单通道
            if len(img_slice.shape) == 2:
                img_slice = img_slice[np.newaxis, ...]  # 添加通道维度 [1, H, W]
            
            # 归一化图像
            img_slice = normalize_image(img_slice)
            
            # 确保标签是正确的类型
            label_slice = label_slice.astype(np.uint8)
            
            slices_data.append({
                'image': img_slice,  # [1, H, W]
                'label': label_slice,  # [H, W]
                'slice_idx': slice_idx
            })
    
    return slices_data


def process_patient_data(patient_dir, patient_id):
    """处理单个患者的数据"""
    print(f"处理患者: {patient_id}")
    
    # 查找图像和标签文件
    img_files = []
    label_files = []
    
    for file in os.listdir(patient_dir):
        if file.endswith('.nii.gz'):
            if 'gt' in file.lower():
                label_files.append(file)
            else:
                img_files.append(file)
    
    if not img_files or not label_files:
        print(f"警告: 患者 {patient_id} 缺少必要文件")
        return []
    
    patient_slices = []
    
    # 处理每对图像-标签文件
    for img_file in img_files:
        # 查找对应的标签文件
        base_name = img_file.replace('.nii.gz', '')
        corresponding_label = None
        
        for label_file in label_files:
            if base_name in label_file or ('frame' in img_file and 'frame' in label_file):
                corresponding_label = label_file
                break
        
        if corresponding_label is None:
            print(f"警告: 找不到 {img_file} 对应的标签文件")
            continue
        
        # 加载NIFTI文件
        try:
            img_path = os.path.join(patient_dir, img_file)
            label_path = os.path.join(patient_dir, corresponding_label)
            
            img_nii = nib.load(img_path)
            label_nii = nib.load(label_path)
            
            img_data = img_nii.get_fdata()
            label_data = label_nii.get_fdata()
            
            # 验证数据
            is_valid, message = validate_nifti_data(img_data, label_data)
            if not is_valid:
                print(f"跳过 {img_file}: {message}")
                continue
            
            # 处理不同维度的数据
            if len(img_data.shape) == 4:
                # 4D数据，提取ED和ES帧
                (ed_img, ed_label), (es_img, es_label) = extract_ed_es_frames(img_data, label_data)
                
                # 提取ED帧的2D切片
                ed_slices = extract_2d_slices(ed_img, ed_label)
                for slice_data in ed_slices:
                    slice_data['patient_id'] = patient_id
                    slice_data['frame_type'] = 'ED'
                    slice_data['source_file'] = img_file
                patient_slices.extend(ed_slices)
                
                # 提取ES帧的2D切片
                es_slices = extract_2d_slices(es_img, es_label)
                for slice_data in es_slices:
                    slice_data['patient_id'] = patient_id
                    slice_data['frame_type'] = 'ES'
                    slice_data['source_file'] = img_file
                patient_slices.extend(es_slices)
                
            elif len(img_data.shape) == 3:
                # 3D数据，直接提取2D切片
                slices = extract_2d_slices(img_data, label_data)
                for slice_data in slices:
                    slice_data['patient_id'] = patient_id
                    slice_data['frame_type'] = 'single'
                    slice_data['source_file'] = img_file
                patient_slices.extend(slices)
            
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")
            continue
    
    print(f"患者 {patient_id} 提取了 {len(patient_slices)} 个有效切片")
    return patient_slices


def save_h5_dataset(slices_data, output_path, dataset_name):
    """保存切片数据到H5文件"""
    print(f"保存 {dataset_name} 数据到 {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # 创建数据集
        num_slices = len(slices_data)
        
        # 获取第一个样本的形状来初始化数据集
        sample_img = slices_data[0]['image']
        sample_label = slices_data[0]['label']
        
        img_shape = (num_slices,) + sample_img.shape  # [N, 1, H, W]
        label_shape = (num_slices,) + sample_label.shape  # [N, H, W]
        
        # 创建数据集
        img_dataset = f.create_dataset('image', img_shape, dtype=np.float32)
        label_dataset = f.create_dataset('label', label_shape, dtype=np.uint8)
        
        # 创建元数据数组
        patient_ids = []
        frame_types = []
        slice_indices = []
        source_files = []
        
        # 填充数据
        for i, slice_data in enumerate(tqdm(slices_data, desc="保存数据")):
            img_dataset[i] = slice_data['image']
            label_dataset[i] = slice_data['label']
            
            patient_ids.append(slice_data['patient_id'].encode('utf-8'))
            frame_types.append(slice_data['frame_type'].encode('utf-8'))
            slice_indices.append(slice_data['slice_idx'])
            source_files.append(slice_data['source_file'].encode('utf-8'))
        
        # 保存元数据
        f.create_dataset('patient_id', data=patient_ids)
        f.create_dataset('frame_type', data=frame_types)
        f.create_dataset('slice_idx', data=slice_indices)
        f.create_dataset('source_file', data=source_files)
        
        # 保存数据集统计信息
        f.attrs['num_slices'] = num_slices
        f.attrs['image_shape'] = sample_img.shape
        f.attrs['label_shape'] = sample_label.shape
        f.attrs['num_classes'] = 4  # ACDC: 背景, LV, RV, MYO
        f.attrs['class_names'] = [b'background', b'LV', b'RV', b'MYO']
        
        print(f"数据集统计:")
        print(f"  切片数量: {num_slices}")
        print(f"  图像形状: {sample_img.shape}")
        print(f"  标签形状: {sample_label.shape}")
        print(f"  类别数量: 4")


def create_patient_list_file(output_dir, train_patients, val_patients):
    """创建患者列表文件"""
    train_list_path = os.path.join(output_dir, 'train_patients.txt')
    val_list_path = os.path.join(output_dir, 'val_patients.txt')
    
    with open(train_list_path, 'w') as f:
        for patient in train_patients:
            f.write(f"{patient}\n")
    
    with open(val_list_path, 'w') as f:
        for patient in val_patients:
            f.write(f"{patient}\n")
    
    print(f"患者列表文件已保存:")
    print(f"  训练集: {train_list_path} ({len(train_patients)} 患者)")
    print(f"  验证集: {val_list_path} ({len(val_patients)} 患者)")


def main():
    parser = argparse.ArgumentParser(description='ACDC数据集预处理')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/jovyan/work/medical_dataset/ACDC',
                       help='ACDC原始数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/jovyan/work/medical_dataset/ACDC_processed',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--min_label_pixels', type=int, default=50,
                       help='切片中最少标签像素数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有患者目录
    training_dir = os.path.join(args.input_dir, 'training')
    if not os.path.exists(training_dir):
        print(f"错误: 找不到训练数据目录 {training_dir}")
        return
    
    patient_dirs = []
    for item in os.listdir(training_dir):
        item_path = os.path.join(training_dir, item)
        if os.path.isdir(item_path) and item.startswith('patient'):
            patient_dirs.append((item, item_path))
    
    patient_dirs.sort()  # 按患者ID排序
    print(f"找到 {len(patient_dirs)} 个患者目录")
    
    # 处理所有患者数据
    all_slices = []
    patient_slice_counts = {}
    
    for patient_id, patient_path in tqdm(patient_dirs, desc="处理患者数据"):
        patient_slices = process_patient_data(patient_path, patient_id)
        all_slices.extend(patient_slices)
        patient_slice_counts[patient_id] = len(patient_slices)
    
    print(f"\n总共提取了 {len(all_slices)} 个有效切片")
    
    # 按患者划分训练集和验证集
    patient_ids = list(patient_slice_counts.keys())
    train_patients, val_patients = train_test_split(
        patient_ids, train_size=args.train_ratio, random_state=args.seed
    )
    
    # 分离训练集和验证集的切片
    train_slices = []
    val_slices = []
    
    for slice_data in all_slices:
        if slice_data['patient_id'] in train_patients:
            train_slices.append(slice_data)
        else:
            val_slices.append(slice_data)
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_patients)} 患者, {len(train_slices)} 切片")
    print(f"  验证集: {len(val_patients)} 患者, {len(val_slices)} 切片")
    
    # 保存训练集
    if train_slices:
        train_h5_path = os.path.join(args.output_dir, 'train.h5')
        save_h5_dataset(train_slices, train_h5_path, 'train')
    
    # 保存验证集
    if val_slices:
        val_h5_path = os.path.join(args.output_dir, 'val.h5')
        save_h5_dataset(val_slices, val_h5_path, 'val')
    
    # 创建患者列表文件
    create_patient_list_file(args.output_dir, train_patients, val_patients)
    
    # 保存处理参数
    params_path = os.path.join(args.output_dir, 'preprocessing_params.txt')
    with open(params_path, 'w') as f:
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Train ratio: {args.train_ratio}\n")
        f.write(f"Min label pixels: {args.min_label_pixels}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Total patients: {len(patient_dirs)}\n")
        f.write(f"Total slices: {len(all_slices)}\n")
        f.write(f"Train patients: {len(train_patients)}\n")
        f.write(f"Val patients: {len(val_patients)}\n")
        f.write(f"Train slices: {len(train_slices)}\n")
        f.write(f"Val slices: {len(val_slices)}\n")
    
    print(f"\n预处理完成！参数已保存到 {params_path}")
    print(f"数据已保存到 {args.output_dir}")


if __name__ == '__main__':
    main() 