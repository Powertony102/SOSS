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
from scipy import ndimage
from skimage import transform


def validate_nifti_data(img_data, label_data):
    """验证NIFTI数据的有效性"""
    if img_data is None or label_data is None:
        return False
    if img_data.shape != label_data.shape:
        print(f"警告: 图像和标签形状不匹配: {img_data.shape} vs {label_data.shape}")
        return False
    if np.isnan(img_data).any() or np.isinf(img_data).any():
        print("警告: 图像包含NaN或Inf值")
        return False
    return True


def extract_ed_es_frames(img_4d, label_4d):
    """提取ED和ES帧"""
    # 通常ED是第一帧，ES是中间帧（根据心室体积判断）
    if img_4d.ndim == 4:
        # 计算每一帧的标签体积来判断ED/ES
        volumes = []
        for t in range(img_4d.shape[3]):
            label_frame = label_4d[:, :, :, t]
            # 计算左心室(LV=3)的体积
            lv_volume = np.sum(label_frame == 3)
            volumes.append(lv_volume)
        
        # ED: 最大体积（舒张末期）
        # ES: 最小体积（收缩末期）
        ed_idx = np.argmax(volumes)
        es_idx = np.argmin(volumes)
        
        return ed_idx, es_idx
    return 0, 0


def normalize_image(image):
    """图像强度归一化到[0,1]"""
    # 去除背景（值为0的区域）计算统计量
    foreground_mask = image > 0
    if np.sum(foreground_mask) > 0:
        mean_val = np.mean(image[foreground_mask])
        std_val = np.std(image[foreground_mask])
        if std_val > 0:
            # Z-score归一化
            image = (image - mean_val) / std_val
            # 截断到[-3, 3]范围
            image = np.clip(image, -3, 3)
            # 归一化到[0, 1]
            image = (image + 3) / 6
        else:
            image = image / (np.max(image) + 1e-8)
    return image.astype(np.float32)


def resize_image_and_label(image, label, target_size=(256, 256)):
    """将图像和标签resize到目标尺寸
    
    Args:
        image: 输入图像 [H, W] 
        label: 输入标签 [H, W]
        target_size: 目标尺寸 (H, W)
    
    Returns:
        resized_image, resized_label
    """
    # 图像使用双线性插值
    resized_image = transform.resize(
        image, 
        target_size, 
        order=1,  # 双线性插值
        preserve_range=True,
        anti_aliasing=True,
        mode='constant'
    ).astype(np.float32)
    
    # 标签使用最近邻插值
    resized_label = transform.resize(
        label, 
        target_size, 
        order=0,  # 最近邻插值
        preserve_range=True,
        anti_aliasing=False,
        mode='constant'
    ).astype(np.uint8)
    
    return resized_image, resized_label


def extract_2d_slices(img_3d, label_3d, min_label_pixels=50, target_size=(256, 256)):
    """从3D图像中提取有效的2D切片并resize到目标尺寸"""
    slices = []
    
    for z in range(img_3d.shape[2]):
        img_slice = img_3d[:, :, z]
        label_slice = label_3d[:, :, z]
        
        # 检查是否包含足够的标签像素
        if np.sum(label_slice > 0) >= min_label_pixels:
            # 归一化图像
            img_slice = normalize_image(img_slice)
            
            # Resize到目标尺寸
            img_slice_resized, label_slice_resized = resize_image_and_label(
                img_slice, label_slice, target_size
            )
            
            # 添加通道维度 [H, W] -> [1, H, W]
            img_slice_resized = img_slice_resized[None, ...]
            
            slices.append({
                'image': img_slice_resized,
                'label': label_slice_resized,
                'slice_idx': z
            })
    
    return slices


def process_patient_data(patient_dir, patient_id, target_size=(256, 256)):
    """处理单个患者的数据"""
    slices = []
    
    # 查找ED和ES帧文件
    frame_files = []
    for filename in os.listdir(patient_dir):
        if filename.endswith('.nii.gz') and 'frame' in filename and '_gt' not in filename:
            frame_files.append(filename)
    
    frame_files.sort()
    
    for frame_file in frame_files:
        frame_path = os.path.join(patient_dir, frame_file)
        
        # 构造对应的标签文件路径
        label_file = frame_file.replace('.nii.gz', '_gt.nii.gz')
        label_path = os.path.join(patient_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"警告: 找不到标签文件 {label_path}")
            continue
        
        try:
            # 加载NIFTI文件
            img_nii = nib.load(frame_path)
            label_nii = nib.load(label_path)
            
            img_data = img_nii.get_fdata()
            label_data = label_nii.get_fdata().astype(np.uint8)
            
            # 验证数据
            if not validate_nifti_data(img_data, label_data):
                continue
            
            # 确定帧类型
            if 'frame01' in frame_file:
                frame_type = 'ED'
            elif 'frame' in frame_file:
                frame_type = 'ES'
            else:
                frame_type = 'UNK'
            
            # 提取2D切片
            patient_slices = extract_2d_slices(img_data, label_data, target_size=target_size)
            
            # 添加元数据
            for slice_data in patient_slices:
                slice_data.update({
                    'patient_id': patient_id,
                    'frame_type': frame_type,
                    'source_file': frame_file
                })
                slices.append(slice_data)
            
            print(f"  {frame_file}: 提取了 {len(patient_slices)} 个切片")
            
        except Exception as e:
            print(f"处理 {frame_file} 时出错: {e}")
            continue
    
    return slices


def save_h5_dataset(slices_data, output_path, dataset_name, target_size=(256, 256)):
    """保存切片数据到H5文件"""
    if not slices_data:
        print(f"警告: {dataset_name} 数据集为空，跳过保存")
        return
    
    print(f"保存 {dataset_name} 数据到 {output_path}")
    
    num_slices = len(slices_data)
    
    with h5py.File(output_path, 'w') as f:
        # 使用固定的图像形状
        img_shape = (num_slices, 1, target_size[0], target_size[1])  # [N, 1, H, W]
        label_shape = (num_slices, target_size[0], target_size[1])     # [N, H, W]
        
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
            # 确保图像形状正确
            img = slice_data['image']
            if img.shape != (1, target_size[0], target_size[1]):
                print(f"警告: 切片 {i} 图像形状不匹配: {img.shape}, 期望: {(1, target_size[0], target_size[1])}")
                # 尝试修复形状
                if img.ndim == 2:
                    img = img[None, ...]  # 添加通道维度
                if img.shape[1:] != target_size:
                    img_2d = img[0] if img.ndim == 3 else img
                    img_2d, _ = resize_image_and_label(img_2d, slice_data['label'], target_size)
                    img = img_2d[None, ...]
            
            # 确保标签形状正确
            label = slice_data['label']
            if label.shape != target_size:
                print(f"警告: 切片 {i} 标签形状不匹配: {label.shape}, 期望: {target_size}")
                _, label = resize_image_and_label(slice_data['image'][0] if slice_data['image'].ndim == 3 else slice_data['image'], label, target_size)
            
            img_dataset[i] = img
            label_dataset[i] = label
            
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
        f.attrs['image_shape'] = (1, target_size[0], target_size[1])
        f.attrs['label_shape'] = target_size
        f.attrs['num_classes'] = 4  # ACDC: 背景, LV, RV, MYO
        f.attrs['class_names'] = [b'background', b'LV', b'RV', b'MYO']
        f.attrs['target_size'] = target_size
        
        print(f"数据集统计:")
        print(f"  切片数量: {num_slices}")
        print(f"  图像形状: {(1, target_size[0], target_size[1])}")
        print(f"  标签形状: {target_size}")
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
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='目标图像尺寸 [height, width]')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    target_size = tuple(args.target_size)
    print(f"目标图像尺寸: {target_size}")
    
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
        print(f"处理患者: {patient_id}")
        patient_slices = process_patient_data(patient_path, patient_id, target_size)
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
        save_h5_dataset(train_slices, train_h5_path, 'train', target_size)
    
    # 保存验证集
    if val_slices:
        val_h5_path = os.path.join(args.output_dir, 'val.h5')
        save_h5_dataset(val_slices, val_h5_path, 'val', target_size)
    
    # 保存纯标注数据（用于监督损失）
    if labeled_slices:
        labeled_h5_path = os.path.join(args.output_dir, 'labeled.h5')
        save_h5_dataset(labeled_slices, labeled_h5_path, 'labeled', target_size)
    
    # 创建患者列表文件
    create_patient_list_file(args.output_dir, labeled_patients + unlabeled_patients, val_patients)
    
    print(f"\n预处理完成!")
    print(f"输出目录: {args.output_dir}")
    print(f"训练集: {len(train_slices)} 切片")
    print(f"验证集: {len(val_slices)} 切片") 
    print(f"标注数据: {len(labeled_slices)} 切片")
    print(f"图像尺寸: {target_size}")


if __name__ == "__main__":
    main() 