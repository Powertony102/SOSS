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
        print(f"  图像文件: {img_files}")
        print(f"  标签文件: {label_files}")
        return []
    
    patient_slices = []
    
    # ACDC数据集的特殊处理
    # 根据ACDC数据集结构，每个患者通常有：
    # - patient001_4d.nii.gz (4D时间序列)
    # - patient001_frame01.nii.gz (ED帧)
    # - patient001_frame01_gt.nii.gz (ED帧标签)
    # - patient001_frame??.nii.gz (ES帧)
    # - patient001_frame??_gt.nii.gz (ES帧标签)
    
    # 优先处理frame文件（ED/ES帧）
    frame_files = [f for f in img_files if 'frame' in f]
    
    if frame_files:
        # 处理frame文件
        for img_file in frame_files:
            # 构造对应的标签文件名
            base_name = img_file.replace('.nii.gz', '')
            corresponding_label = f"{base_name}_gt.nii.gz"
            
            if corresponding_label in label_files:
                # 找到对应的标签文件
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
                    
                    # 确定帧类型
                    frame_type = 'ES' if 'frame01' not in img_file else 'ED'
                    
                    # 处理3D数据
                    if len(img_data.shape) == 3:
                        slices = extract_2d_slices(img_data, label_data)
                        for slice_data in slices:
                            slice_data['patient_id'] = patient_id
                            slice_data['frame_type'] = frame_type
                            slice_data['source_file'] = img_file
                        patient_slices.extend(slices)
                    
                except Exception as e:
                    print(f"处理 {img_file} 时出错: {e}")
                    continue
            else:
                print(f"警告: 找不到 {img_file} 对应的标签文件 {corresponding_label}")
    
    # 如果没有frame文件，处理4D文件
    elif any('4d' in f for f in img_files):
        for img_file in img_files:
            if '4d' in img_file:
                # 4D文件通常没有直接对应的标签，跳过或特殊处理
                print(f"跳过4D文件 {img_file}（通常用于生成frame文件）")
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


def create_semisupervised_split(slices_data, labeled_ratio=0.05, val_ratio=0.2, seed=42):
    """创建半监督学习的数据划分
    
    Args:
        slices_data: 所有切片数据
        labeled_ratio: 标注数据比例（默认5%）
        val_ratio: 验证集比例（默认20%）
        seed: 随机种子
    
    Returns:
        labeled_slices, unlabeled_slices, val_slices
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 按患者分组
    patient_slices = {}
    for slice_data in slices_data:
        patient_id = slice_data['patient_id']
        if patient_id not in patient_slices:
            patient_slices[patient_id] = []
        patient_slices[patient_id].append(slice_data)
    
    patient_ids = list(patient_slices.keys())
    random.shuffle(patient_ids)
    
    total_patients = len(patient_ids)
    val_patients_count = int(total_patients * val_ratio)
    labeled_patients_count = int((total_patients - val_patients_count) * labeled_ratio)
    
    # 划分患者
    val_patients = patient_ids[:val_patients_count]
    train_patients = patient_ids[val_patients_count:]
    labeled_patients = train_patients[:labeled_patients_count]
    unlabeled_patients = train_patients[labeled_patients_count:]
    
    # 收集切片
    labeled_slices = []
    unlabeled_slices = []
    val_slices = []
    
    for patient_id in labeled_patients:
        labeled_slices.extend(patient_slices[patient_id])
    
    for patient_id in unlabeled_patients:
        unlabeled_slices.extend(patient_slices[patient_id])
    
    for patient_id in val_patients:
        val_slices.extend(patient_slices[patient_id])
    
    print(f"\n半监督数据划分:")
    print(f"  标注数据: {len(labeled_patients)} 患者, {len(labeled_slices)} 切片 ({len(labeled_slices)/len(slices_data)*100:.1f}%)")
    print(f"  未标注数据: {len(unlabeled_patients)} 患者, {len(unlabeled_slices)} 切片 ({len(unlabeled_slices)/len(slices_data)*100:.1f}%)")
    print(f"  验证数据: {len(val_patients)} 患者, {len(val_slices)} 切片 ({len(val_slices)/len(slices_data)*100:.1f}%)")
    
    return labeled_slices, unlabeled_slices, val_slices, labeled_patients, unlabeled_patients, val_patients


def main():
    parser = argparse.ArgumentParser(description='ACDC数据集预处理')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/jovyan/work/medical_dataset/ACDC',
                       help='ACDC原始数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/jovyan/work/medical_dataset/ACDC_processed',
                       help='输出目录')
    parser.add_argument('--labeled_ratio', type=float, default=0.05,
                       help='标注数据比例（半监督学习）')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
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
    
    # 进行半监督学习的数据划分
    labeled_slices, unlabeled_slices, val_slices, labeled_patients, unlabeled_patients, val_patients = create_semisupervised_split(
        all_slices, args.labeled_ratio, args.val_ratio, args.seed
    )
    
    # 合并训练数据（标注 + 未标注）
    train_slices = labeled_slices + unlabeled_slices
    
    # 保存训练集（包含标注和未标注数据）
    if train_slices:
        train_h5_path = os.path.join(args.output_dir, 'train.h5')
        save_h5_dataset(train_slices, train_h5_path, 'train')
    
    # 保存验证集
    if val_slices:
        val_h5_path = os.path.join(args.output_dir, 'val.h5')
        save_h5_dataset(val_slices, val_h5_path, 'val')
    
    # 保存纯标注数据（用于监督损失）
    if labeled_slices:
        labeled_h5_path = os.path.join(args.output_dir, 'labeled.h5')
        save_h5_dataset(labeled_slices, labeled_h5_path, 'labeled')
    
    # 创建患者列表文件
    create_patient_list_file(args.output_dir, labeled_patients + unlabeled_patients, val_patients)
    
    # 创建半监督学习的划分文件
    labeled_list_path = os.path.join(args.output_dir, 'labeled_patients.txt')
    unlabeled_list_path = os.path.join(args.output_dir, 'unlabeled_patients.txt')
    
    with open(labeled_list_path, 'w') as f:
        for patient in labeled_patients:
            f.write(f"{patient}\n")
    
    with open(unlabeled_list_path, 'w') as f:
        for patient in unlabeled_patients:
            f.write(f"{patient}\n")
    
    # 保存切片级别的索引文件
    slice_indices_path = os.path.join(args.output_dir, 'slice_indices.txt')
    with open(slice_indices_path, 'w') as f:
        f.write(f"labeled_slices_count: {len(labeled_slices)}\n")
        f.write(f"unlabeled_slices_start: {len(labeled_slices)}\n")
        f.write(f"unlabeled_slices_count: {len(unlabeled_slices)}\n")
        f.write(f"total_train_slices: {len(train_slices)}\n")
    
    # 保存处理参数
    params_path = os.path.join(args.output_dir, 'preprocessing_params.txt')
    with open(params_path, 'w') as f:
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Labeled ratio: {args.labeled_ratio}\n")
        f.write(f"Val ratio: {args.val_ratio}\n")
        f.write(f"Min label pixels: {args.min_label_pixels}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Total patients: {len(patient_dirs)}\n")
        f.write(f"Total slices: {len(all_slices)}\n")
        f.write(f"Labeled patients: {len(labeled_patients)}\n")
        f.write(f"Unlabeled patients: {len(unlabeled_patients)}\n")
        f.write(f"Val patients: {len(val_patients)}\n")
        f.write(f"Labeled slices: {len(labeled_slices)}\n")
        f.write(f"Unlabeled slices: {len(unlabeled_slices)}\n")
        f.write(f"Train slices: {len(train_slices)}\n")
        f.write(f"Val slices: {len(val_slices)}\n")
    
    print(f"\n预处理完成！参数已保存到 {params_path}")
    print(f"数据已保存到 {args.output_dir}")
    print(f"\n文件结构:")
    print(f"├── train.h5                    # 训练集（标注+未标注）")
    print(f"├── labeled.h5                  # 纯标注数据")
    print(f"├── val.h5                      # 验证集")
    print(f"├── labeled_patients.txt        # 标注患者列表")
    print(f"├── unlabeled_patients.txt      # 未标注患者列表")
    print(f"├── slice_indices.txt           # 切片索引信息")
    print(f"└── preprocessing_params.txt    # 预处理参数")


if __name__ == '__main__':
    main() 