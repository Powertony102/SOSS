# 类间分离模块 (Inter-Class Separation via Prototypes)

## 概述

本实现基于您提供的文档描述，参考了 SS-Net 的设计思路，在现有的 CORN-DFP 框架中集成了类间分离模块。该模块通过原型（prototype）引导的特征聚类策略，将同类像素特征拉近、异类特征拉远，有效应对类间特征相似的挑战。

## 主要特性

### 1. 原型选择与管理
- **高质量原型选择**: 从标注数据中选择高置信度的像素特征作为原型候选
- **智能筛选**: 综合考虑置信度和特征范数，选择前K个最佳候选原型
- **背景类忽略**: 自动忽略背景类，避免背景像素对原型计算的干扰

### 2. 在线更新机制
- **滑动平均更新**: 使用滑动平均策略在线更新原型，适应模型训练过程中的特征变化
- **动态调整**: 原型随着训练进度不断反映高质量特征的位置变化

### 3. 原型分离损失
- **类内紧致损失**: $L_{intra} = \frac{1}{N}\sum_i |f_i - \mu_{y_i}|^2$，拉近同类像素特征
- **类间分离损失**: $L_{inter} = \frac{1}{C(C-1)} \sum_{c\neq c'} \max(0, m - | \mu_c - \mu_{c'} |)^2$，推远异类原型

### 4. 半监督学习集成
- **标注数据**: 提供高质量原型指导，使用真实标签更新原型
- **未标注数据**: 使用伪标签计算损失，仅在高置信度区域应用
- **置信度过滤**: 自动过滤低置信度预测，减少错误伪标签的干扰

## 文件结构

```
myutils/
├── prototype_manager.py      # 原型管理器核心实现
├── losses.py                 # 现有损失函数
└── ...

train_cov_dfp_3d.py          # 修改后的训练脚本
run_prototype_separation.sh   # 使用示例脚本
README_Prototype_Separation.md # 本文档
```

## 使用方法

### 1. 基本使用

```bash
# 仅使用原型分离功能
python train_cov_dfp_3d.py \
    --exp corn_prototype \
    --dataset_name LA \
    --dataset_path /path/to/dataset \
    --use_prototype \
    --prototype_confidence_threshold 0.8 \
    --prototype_k 10 \
    --prototype_intra_weight 1.0 \
    --prototype_inter_weight 0.1 \
    --prototype_margin 1.0
```

### 2. 与DFP结合使用

```bash
# 原型分离 + DFP组合
python train_cov_dfp_3d.py \
    --exp corn_prototype_dfp \
    --dataset_name LA \
    --dataset_path /path/to/dataset \
    --use_prototype \
    --use_dfp \
    --prototype_confidence_threshold 0.8 \
    --prototype_k 10 \
    --prototype_intra_weight 1.0 \
    --prototype_inter_weight 0.1 \
    --prototype_margin 1.0 \
    --embedding_dim 64 \
    --memory_num 256
```

### 3. 使用提供的脚本

```bash
# 运行示例脚本
./run_prototype_separation.sh
```

## 参数说明

### 原型分离参数
- `--use_prototype`: 是否启用原型分离功能
- `--prototype_confidence_threshold`: 置信度阈值（默认0.8）
- `--prototype_k`: 每类选择的候选原型数量（默认10）
- `--prototype_update_momentum`: 滑动平均更新动量（默认0.9）
- `--prototype_intra_weight`: 类内紧致损失权重（默认1.0）
- `--prototype_inter_weight`: 类间分离损失权重（默认0.1）
- `--prototype_margin`: 类间分离的最小距离（默认1.0）
- `--prototype_update_freq`: 原型更新频率（默认1，每次迭代）

### 推荐参数设置
- **置信度阈值**: 0.8-0.9，平衡质量和数量
- **类内权重**: 1.0，主要目标
- **类间权重**: 0.1，辅助约束
- **分离边界**: 1.0，根据特征空间调整

## 监控与日志

### Wandb记录
- `train/loss_prototype`: 原型分离总损失
- `train/lambda_p`: 原型损失权重
- `train/intra_loss_labeled`: 标注数据类内损失
- `train/inter_loss_labeled`: 标注数据类间损失
- `train/intra_loss_unlabeled`: 未标注数据类内损失
- `train/inter_loss_unlabeled`: 未标注数据类间损失

### 训练日志
```
Iteration 1000 : loss : 0.750, loss_s: 0.425, loss_c: 0.235, loss_prototype: 0.090
```

## 效果与优势

### 1. 特征空间优化
- 形成清晰的簇结构，每个类别在嵌入空间收敛到其原型附近
- 类别间分布更紧凑且彼此远离
- 提高对强度相似但解剖学不同结构的区分能力

### 2. 半监督学习增强
- 标注数据提供高质量原型指导
- 未标注数据在原型引力作用下获得额外监督信号
- 逐步纠正特征表示，减少高熵（不确定）区域

### 3. 鲁棒性提升
- 自动过滤低置信度预测，减少错误信号影响
- 动态更新机制适应训练过程中的模型变化
- 与现有DFP框架无缝集成

## 技术细节

### 原型计算
```python
# 初始原型（平均值）
prototype = torch.mean(high_confidence_features, dim=0)

# 滑动平均更新
updated_prototype = momentum * old_prototype + (1 - momentum) * new_prototype
```

### 损失计算
```python
# 类内紧致损失
intra_loss = torch.mean((features - prototype) ** 2)

# 类间分离损失
inter_loss = torch.max(0, margin - distance) ** 2
```

## 实验建议

1. **消融研究**: 分别测试类内和类间损失的贡献
2. **参数调优**: 调整置信度阈值和权重比例
3. **数据集适应**: 根据不同数据集调整原型数量和更新策略
4. **性能对比**: 与基线模型和其他对比学习方法比较

## 故障排除

### 常见问题
1. **原型未初始化**: 确保有足够的高置信度样本
2. **GPU内存不足**: 减少批次大小或特征维度
3. **损失不收敛**: 调整权重平衡和学习率

### 调试建议
- 监控原型初始化状态和更新频率
- 观察类间距离统计信息
- 检查高置信度样本的数量和分布

---

该实现忠实遵循了您文档中的设计理念，结合了SS-Net的成功经验，并与现有的CORN-DFP框架完美集成。通过原型引导的特征聚类策略，有效解决了类间特征相似的挑战，为半监督医学图像分割提供了强有力的工具。 