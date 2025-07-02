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

class ACDCDataSet(Dataset):
    """ACDC数据集加载器
    
    支持从原始NIFTI文件或预处理的H5文件加载ACDC数据
    """
    def __init__(self, base_dir=None, split='train', num=None, transform=None, 
                 use_h5=True, with_idx=False):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.use_h5 = use_h5  # 是否使用预处理的H5文件
        self.with_idx = with_idx
        self.sample_list = []
        
        if self.use_h5:
            self._load_h5_samples(num)
        else:
            self._load_nifti_samples(num)
            
        print("ACDC数据集加载完成，总共 {} 个样本".format(len(self.sample_list)))

    def _load_h5_samples(self, num):
        """从预处理的H5文件加载样本"""
        if self.split == 'train':
            # 从训练切片文件夹加载
            h5_files = sorted(glob(os.path.join(self._base_dir, "slices", "*.h5")))
            self.sample_list = [os.path.basename(f).replace('.h5', '') for f in h5_files]
        else:
            # 从验证列表加载
            with open(os.path.join(self._base_dir, 'val.list'), 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]

    def _load_nifti_samples(self, num):
        """从原始NIFTI文件加载样本"""
        if self.split == 'train':
            # 扫描training文件夹
            patient_dirs = sorted(glob(os.path.join(self._base_dir, "training", "patient*")))
            for patient_dir in patient_dirs:
                patient_id = os.path.basename(patient_dir)
                # 获取ED和ES帧
                nii_files = glob(os.path.join(patient_dir, f"{patient_id}_frame*.nii.gz"))
                # 排除4d文件和gt文件
                nii_files = [f for f in nii_files if "4d" not in f and "gt" not in f]
                for nii_file in nii_files:
                    frame_name = os.path.basename(nii_file).replace('.nii.gz', '')
                    self.sample_list.append((patient_dir, frame_name))
        else:
            # 测试集处理类似
            patient_dirs = sorted(glob(os.path.join(self._base_dir, "testing", "patient*")))
            for patient_dir in patient_dirs:
                patient_id = os.path.basename(patient_dir)
                nii_files = glob(os.path.join(patient_dir, f"{patient_id}_frame*.nii.gz"))
                nii_files = [f for f in nii_files if "4d" not in f and "gt" not in f]
                for nii_file in nii_files:
                    frame_name = os.path.basename(nii_file).replace('.nii.gz', '')
                    self.sample_list.append((patient_dir, frame_name))
        
        if num is not None:
            self.sample_list = self.sample_list[:num]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.use_h5:
            return self._get_h5_item(idx)
        else:
            return self._get_nifti_item(idx)

    def _get_h5_item(self, idx):
        """从H5文件获取样本"""
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(os.path.join(self._base_dir, "slices", f"{case}.h5"), 'r')
        else:
            h5f = h5py.File(os.path.join(self._base_dir, f"{case}.h5"), 'r')
        
        image = h5f['image'][:]
        label = h5f['label'][:]
        h5f.close()
        
        sample = {'image': image, 'label': label}
        if self.split == "train" and self.transform:
            sample = self.transform(sample)
        
        if self.with_idx:
            sample["idx"] = idx
        return sample

    def _get_nifti_item(self, idx):
        """从NIFTI文件获取样本"""
        patient_dir, frame_name = self.sample_list[idx]
        
        # 读取图像
        image_path = os.path.join(patient_dir, f"{frame_name}.nii.gz")
        img_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(img_itk)
        
        # 读取标签
        label_path = os.path.join(patient_dir, f"{frame_name}_gt.nii.gz")
        if os.path.exists(label_path):
            lbl_itk = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(lbl_itk)
        else:
            # 如果没有标签，创建零标签
            label = np.zeros_like(image)
        
        # 归一化图像
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = image.astype(np.float32)
        
        # 随机选择一个切片（如果是3D体积）
        if len(image.shape) == 3 and image.shape[0] > 1:
            slice_idx = random.randint(0, image.shape[0] - 1)
            image = image[slice_idx]
            label = label[slice_idx]
        
        sample = {'image': image, 'label': label}
        if self.split == "train" and self.transform:
            sample = self.transform(sample)
        
        if self.with_idx:
            sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    """随机旋转和翻转"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
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
    """ACDC数据增强生成器"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # 随机数据增强
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # 调整大小
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 转换为张量
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
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