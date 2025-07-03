# SOSS:  Second Order Semi-Supervised Segmentation for Medical Image



## Introduction

This repository is for our paper: *SOSS: Second Order Semi-Supervised Segmentation for Medical Image*

## Hierarchical Covariance Consistency (HCC)

### 概述

分层协方差一致性 (Hierarchical Covariance Consistency, HCC) 是一种新的正则化技术，通过在编码器的多个层级上强制特征协方差矩阵的一致性来提升模型性能。

### 核心机制

HCC 损失在网络的多个编码器层级上计算特征协方差矩阵，并使用 CORAL 损失来最小化双分支网络对应层级间的协方差差异：

```
L_HCC = Σ(l=1 to L) w_l × L_CORAL(C_S^l, C_T^l)
```

其中：
- `L` 是监督的总层级数 (默认5层)
- `w_l` 是第 `l` 层的权重系数
- `C_S^l` 和 `C_T^l` 分别是两个分支网络在第 `l` 层的特征协方差矩阵

### 理论优势

1. **多尺度结构约束**：浅层特征捕捉局部结构，深层特征捕捉全局语义
2. **隐式形状先验**：引导模型生成更规整、符合解剖学常识的分割结果
3. **结构一致性**：奖励结构级一致性而非像素级对齐

### 使用方法

#### 命令行参数

```bash
# 启用 HCC 损失
python train_corn_3d.py --lambda_hcc 0.1

# 自定义层级权重 (5个权重对应5个编码器层)
python train_corn_3d.py --lambda_hcc 0.1 --hcc_weights "0.5,0.5,1,1,1.5"

# 结合其他功能
python train_corn_3d.py --use_wandb --lambda_hcc 0.1 --cov_mode patch --patch_size 4
```

#### 参数说明

- `--lambda_hcc`: HCC 损失的权重系数 (默认: 0.1)
- `--hcc_weights`: 5个编码器层的权重，用逗号分隔 (默认: "0.5,0.5,1,1,1.5")
- `--cov_mode`: 协方差计算模式，'patch' 或 'full' (默认: 'patch')
- `--patch_size`: patch 大小，当 cov_mode='patch' 时使用 (默认: 4)

#### 推荐配置

```bash
# 基础配置
python train_corn_3d.py --lambda_hcc 0.1 --hcc_weights "0.5,0.5,1,1,1.5"

# 高性能配置（更注重深层特征）
python train_corn_3d.py --lambda_hcc 0.15 --hcc_weights "0.3,0.5,0.8,1.2,1.5"

# 轻量配置（更均匀的权重）
python train_corn_3d.py --lambda_hcc 0.05 --hcc_weights "1,1,1,1,1"
```

### 实现细节

HCC 损失支持：
- 3D 医学图像分割 (5D 张量: [B,C,D,H,W])
- 2D 图像分割 (4D 张量: [B,C,H,W])
- Patch-wise 和全局协方差计算模式
- 与现有 CORAL 损失和其他正则化技术的兼容

### 监控指标

训练过程中可以通过以下指标监控 HCC 损失：
- `train/loss_hcc`: HCC 损失值
- `train/lambda_hcc`: HCC 损失权重
- 各层级的协方差差异（通过 wandb 可视化）
