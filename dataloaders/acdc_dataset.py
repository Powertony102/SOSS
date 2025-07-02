import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import SimpleITK as sitk
from torchvision import transforms

class ACDCDataSet(Dataset):
    """ACDC数据集加载器 - 优化版本"""
    
    def __init__(self, base_dir, list_dir, split, transform=None):
        """
        Args:
            base_dir: 数据基础目录
            list_dir: 列表文件目录（可以为None，自动从base_dir查找）
            split: 'train' 或 'val'
            transform: 数据变换
        """
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        
        # 直接从H5文件加载数据
        h5_file_path = os.path.join(base_dir, f'{split}.h5')
        
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"找不到H5文件: {h5_file_path}")
        
        print(f"加载 {split} 数据集: {h5_file_path}")
        
        # 打开H5文件并读取数据
        with h5py.File(h5_file_path, 'r') as f:
            # 读取图像和标签数据
            self.images = f['image'][:]  # [N, 1, H, W]
            self.labels = f['label'][:]  # [N, H, W]
            
            # 读取元数据（如果需要）
            if 'patient_id' in f:
                self.patient_ids = [pid.decode('utf-8') for pid in f['patient_id'][:]]
            else:
                self.patient_ids = [f"unknown_{i}" for i in range(len(self.images))]
                
            if 'frame_type' in f:
                self.frame_types = [ft.decode('utf-8') for ft in f['frame_type'][:]]
            else:
                self.frame_types = ["unknown"] * len(self.images)
            
            if 'slice_idx' in f:
                self.slice_indices = f['slice_idx'][:]
            else:
                self.slice_indices = list(range(len(self.images)))
            
            # 读取数据集属性
            self.num_classes = f.attrs.get('num_classes', 4)
            
        print(f"数据集加载完成:")
        print(f"  样本数量: {len(self.images)}")
        print(f"  图像形状: {self.images.shape}")
        print(f"  标签形状: {self.labels.shape}")
        print(f"  类别数量: {self.num_classes}")
        
        # 验证数据格式
        self._validate_data_format()
    
    def _validate_data_format(self):
        """验证数据格式是否正确"""
        # 检查图像格式
        if len(self.images.shape) != 4:
            raise ValueError(f"图像数据应该是4D [N, C, H, W]，实际是: {self.images.shape}")
        
        if self.images.shape[1] != 1:
            print(f"警告: 图像不是单通道，形状: {self.images.shape}")
        
        # 检查标签格式
        if len(self.labels.shape) != 3:
            raise ValueError(f"标签数据应该是3D [N, H, W]，实际是: {self.labels.shape}")
        
        # 检查标签值范围
        unique_labels = np.unique(self.labels)
        if np.max(unique_labels) >= self.num_classes:
            print(f"警告: 标签值超出范围，唯一值: {unique_labels}, 期望范围: [0, {self.num_classes-1}]")
        
        print("数据格式验证通过")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 获取图像和标签
        image = self.images[idx].astype(np.float32)  # [1, H, W]
        label = self.labels[idx].astype(np.uint8)    # [H, W]
        
        # 确保图像格式正确
        if image.shape[0] != 1:
            print(f"警告: 图像 {idx} 格式异常: {image.shape}")
            if len(image.shape) == 3 and image.shape[-1] == 1:
                image = image[..., 0][np.newaxis, ...]  # [H, W, 1] -> [1, H, W]
            elif len(image.shape) == 2:
                image = image[np.newaxis, ...]  # [H, W] -> [1, H, W]
            else:
                # 多通道转单通道
                image = np.mean(image, axis=0, keepdims=True)
        
        # 创建样本字典
        sample = {
            'image': image,
            'label': label,
            'idx': idx,
            'patient_id': self.patient_ids[idx],
            'frame_type': self.frame_types[idx],
            'slice_idx': self.slice_indices[idx]
        }
        
        # 应用数据变换
        if self.transform:
            sample = self.transform(sample)
        
        # 转换为torch张量（解决负步长问题）
        image_array = sample['image']
        label_array = sample['label']
        
        # 确保数组是连续的，解决负步长问题
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = image_array.copy()
        if not label_array.flags['C_CONTIGUOUS']:
            label_array = label_array.copy()
            
        image = torch.from_numpy(image_array).float()
        label = torch.from_numpy(label_array).long()
        
        # 最终验证格式
        if len(image.shape) != 3 or image.shape[0] != 1:
            print(f"最终图像格式错误: {image.shape}")
            # 强制修正
            if len(image.shape) == 2:
                image = image.unsqueeze(0)  # [H, W] -> [1, H, W]
            elif len(image.shape) == 3 and image.shape[0] != 1:
                if image.shape[-1] == 1:
                    image = image.squeeze(-1).unsqueeze(0)  # [H, W, 1] -> [1, H, W]
                else:
                    image = image.mean(0, keepdim=True)  # 多通道 -> 单通道
        
        if len(label.shape) != 2:
            print(f"最终标签格式错误: {label.shape}")
            # 强制修正
            if len(label.shape) == 3 and label.shape[0] == 1:
                label = label.squeeze(0)  # [1, H, W] -> [H, W]
            elif len(label.shape) == 3 and label.shape[-1] == 1:
                label = label.squeeze(-1)  # [H, W, 1] -> [H, W]
            elif len(label.shape) == 3:
                # one-hot转类别索引
                label = torch.argmax(label, dim=-1)
        
        return {
            'image': image,      # [1, H, W]
            'label': label,      # [H, W]
            'idx': torch.tensor(idx),
            'patient_id': sample['patient_id'],
            'frame_type': sample['frame_type'],
            'slice_idx': torch.tensor(sample['slice_idx'])
        }


def random_rot_flip(image, label):
    """随机旋转和翻转"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k).copy()
    label = np.rot90(label, k).copy()
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """随机旋转"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """数据增强生成器"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # 随机旋转
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            # 这里简化处理，实际可以使用更复杂的旋转
            
        # 随机翻转（使用copy避免负步长）
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
            
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()
        
        # 确保输出格式正确
        # image应该是 [C, H, W] 格式
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]  # [H, W] -> [1, H, W]
        elif len(image.shape) == 3 and image.shape[0] != 1:
            # 如果不是单通道，转换为单通道
            if image.shape[-1] == 1:
                image = image[..., 0]  # [H, W, 1] -> [H, W]
                image = image[np.newaxis, ...]  # [H, W] -> [1, H, W]
            else:
                image = np.mean(image, axis=-1)  # 多通道转单通道
                image = image[np.newaxis, ...]
        
        # label应该是 [H, W] 格式
        if len(label.shape) == 3:
            if label.shape[0] == 1:
                label = label[0, ...]  # [1, H, W] -> [H, W]
            elif label.shape[-1] == 1:
                label = label[..., 0]  # [H, W, 1] -> [H, W]
            else:
                # 如果是one-hot编码，转换为类别索引
                label = np.argmax(label, axis=-1)
        
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(object):
    """双流批采样器，用于半监督学习"""
    
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0
    
    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                  grouper(secondary_iter, self.secondary_batch_size))
        )
    
    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# 用于3D数据的变换
class RandomCrop(object):
    """随机裁剪3D数据"""
    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # 如果需要，进行填充
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class ToTensor(object):
    """转换为张量"""
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()} 