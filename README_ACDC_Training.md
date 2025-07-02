# ACDC数据集训练指南

## 概述

本指南介绍如何使用重新设计的ACDC数据预处理和训练脚本，确保数据格式正确，避免运行时修复问题。

## 步骤1：数据预处理

### 1.1 运行预处理脚本

使用新的优化预处理脚本，确保输出数据格式完全正确：

```bash
# 预处理ACDC数据集
python dataloaders/acdc_preprocessing.py \
    --input_dir /home/jovyan/work/medical_dataset/ACDC \
    --output_dir /home/jovyan/work/medical_dataset/ACDC_processed \
    --train_ratio 0.8 \
    --min_label_pixels 50 \
    --seed 42
```

### 1.2 预处理脚本特点

- **正确的数据格式**：输出图像为`[1, H, W]`，标签为`[H, W]`
- **智能切片提取**：只保留包含标签的有效切片
- **ED/ES帧处理**：从4D数据中提取关键帧
- **数据归一化**：基于非零区域的Z-score归一化
- **患者级别划分**：确保训练/验证集的患者不重叠

### 1.3 输出结构

```
ACDC_processed/
├── train.h5                    # 训练集H5文件
├── val.h5                      # 验证集H5文件
├── train_patients.txt          # 训练患者列表
├── val_patients.txt            # 验证患者列表
└── preprocessing_params.txt    # 预处理参数记录
```

## 步骤2：训练模型

### 2.1 运行训练脚本

使用优化的训练脚本，直接加载正确格式的数据：

```bash
python train_acdc_cov_dfp.py \
    --dataset_path /home/jovyan/work/medical_dataset/ACDC_processed \
    --labelnum 7 \
    --use_dfp \
    --gpu 1 \
    --model corn2d \
    --exp acdc_soss_corn2d \
    --use_wandb \
    --wandb_project SOSS-ACDC
```

### 2.2 训练脚本特点

- **原生2D支持**：使用`corn2d`模型，无需维度转换
- **正确数据加载**：直接从H5文件加载正确格式的数据
- **无运行时修复**：完全避免数据格式警告和修复
- **完整的损失函数**：支持分割损失、一致性损失、HCC损失、度量学习损失

### 2.3 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_path` | 预处理数据目录 | 必须指定 |
| `--model` | 模型类型 | `corn2d` |
| `--labelnum` | 标注样本数量 | `7` |
| `--use_dfp` | 是否使用动态特征池 | `False` |
| `--lambda_hcc` | HCC损失权重 | `0.1` |
| `--use_wandb` | 是否使用wandb日志 | `False` |

## 步骤3：验证数据格式

如果需要验证预处理结果，可以运行：

```python
import h5py
import numpy as np

# 检查训练数据
with h5py.File('/home/jovyan/work/medical_dataset/ACDC_processed/train.h5', 'r') as f:
    print("训练集统计:")
    print(f"  图像形状: {f['image'].shape}")  # 应该是 [N, 1, H, W]
    print(f"  标签形状: {f['label'].shape}")  # 应该是 [N, H, W]
    print(f"  类别数量: {f.attrs['num_classes']}")  # 应该是 4
    print(f"  图像数据类型: {f['image'].dtype}")
    print(f"  标签数据类型: {f['label'].dtype}")
    
    # 检查标签值范围
    unique_labels = np.unique(f['label'][:100])  # 检查前100个样本
    print(f"  标签唯一值: {unique_labels}")  # 应该是 [0, 1, 2, 3]
```

## 步骤4：故障排除

### 4.1 常见问题

1. **找不到H5文件**
   - 确保预处理脚本运行成功
   - 检查`--dataset_path`参数是否正确

2. **数据形状错误**
   - 重新运行预处理脚本
   - 检查输入NIFTI文件是否损坏

3. **内存不足**
   - 减少`--batch_size`参数
   - 减少`--max_samples`参数

### 4.2 调试选项

```bash
# 启用详细日志
export PYTHONPATH=$PYTHONPATH:.
python train_acdc_cov_dfp.py \
    --dataset_path /home/jovyan/work/medical_dataset/ACDC_processed \
    --labelnum 7 \
    --model corn2d \
    --exp debug_acdc \
    --max_iteration 100  # 短期运行用于调试
```

## 技术改进

### 与之前版本的区别

1. **预处理阶段确保正确性**：
   - 之前：运行时检测和修复数据格式问题
   - 现在：预处理时就确保数据格式完全正确

2. **原生2D架构**：
   - 之前：3D模型 + 维度转换
   - 现在：专门的2D模型架构

3. **简化的数据加载**：
   - 之前：复杂的数据验证和修复逻辑
   - 现在：直接加载H5数据，无需修复

4. **更好的性能**：
   - 消除了运行时数据修复的开销
   - 减少了内存使用和处理时间

## 预期结果

使用新的预处理和训练流程，您应该看到：

- **无数据格式警告**：不再出现"数据诊断"、"形状修复"等信息
- **更快的训练速度**：消除了运行时数据修复的开销
- **更稳定的训练**：数据格式一致性更好
- **清晰的日志**：只显示训练相关的重要信息

## 结论

新的预处理和训练流程通过在预处理阶段确保数据格式的正确性，彻底解决了运行时数据修复的问题，提供了更稳定、更高效的训练体验。 