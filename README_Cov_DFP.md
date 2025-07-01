# 基于二阶统计量的动态特征池（Cov-DFP）框架

## 概述

本框架实现了研究方案文档中提出的基于协方差的动态特征池（Covariance Dynamic Feature Pool, Cov-DFP）方法。该方法使用二阶统计量（协方差/相关性矩阵）来构建多个专业化的特征池，并通过Selector网络进行自适应选择。

## 核心创新

1. **二阶统计量驱动的DFP构建**：基于全局特征的协方差矩阵进行特征值分解，提取主要共变模式
2. **三阶段训练策略**：预训练 → DFP构建 → 交替训练
3. **Selector网络**：学习选择最适合的DFP用于不同的输入区域
4. **动态适应机制**：周期性重构DFP以适应特征分布的变化

## 架构组件

### 1. CovarianceDynamicFeaturePool (`myutils/cov_dynamic_feature_pool.py`)

核心的DFP管理类，实现：
- 全局特征池管理
- 协方差矩阵计算和特征值分解
- 多个DFP的构建和维护
- Selector训练目标生成

**关键方法**：
```python
# 添加特征到全局池
dfp.add_to_global_pool(features)

# 构建DFP
success = dfp.build_dfps()

# 生成Selector训练目标
target_labels = dfp.get_dfp_target_labels(region_features)

# 根据Selector预测获取DFP特征
dfp_features = dfp.get_dfp_features(dfp_indices)
```

### 2. DFP Selector网络 (`networks/selector_network.py`)

实现了两种Selector架构：
- **DFPSelector**：简单的MLP架构，适用于特征向量输入
- **AdaptivePoolingSelector**：带自适应池化的卷积架构，适用于特征图输入

### 3. 增强的CORN网络 (`networks/VNet.py`)

在原有的CORN架构基础上集成了Selector组件：
```python
# 创建带Selector的模型
model = corf(n_channels=1, n_classes=2, feat_dim=64, num_dfp=8, use_selector=True)

# 前向传播支持Selector
output_dict = model(input, with_selector=True, region_features=features)
```

## 三阶段训练流程

### 阶段一：初始预训练 (iter < dfp_start_iter)
- **目标**：训练初步的主模型，建立有意义的全局特征池
- **操作**：
  - 正常的监督学习和一致性训练
  - 收集并管理全局特征池 F_global
  - 应用HCC（分层协方差一致性）损失

### 阶段二：DFP构建 (iter = dfp_start_iter)
- **目标**：基于二阶统计量构建多个DFP
- **流程**：
  1. 计算全局特征的协方差矩阵 Σ 和相关性矩阵 R
  2. 对 R 进行特征值分解：R = Q Λ Q^T
  3. 选取前 num_dfp 个特征向量作为主要共变模式
  4. 根据投影将特征分配到不同DFP：j* = argmax(|f_i^T p_j|)
  5. 计算每个DFP的中心特征

### 阶段三：交替训练 (iter > dfp_start_iter)
- **模式A - Selector训练**：
  - 冻结主模型，训练Selector网络
  - 使用DFP中心距离生成训练目标
  - 优化交叉熵损失
- **模式B - 主模型训练**：
  - 冻结Selector，训练主模型
  - 使用Selector预测的DFP进行特征池选择
  - 应用完整的损失函数

## 使用方法

### 1. 训练新模型

```bash
python train_cov_dfp_3d.py \
    --dataset_name LA \
    --dataset_path /path/to/LA/dataset \
    --exp cov_dfp_experiment \
    --num_dfp 8 \
    --dfp_start_iter 2000 \
    --selector_train_iter 50 \
    --embedding_dim 64 \
    --use_wandb \
    --gpu 0
```

### 2. 关键参数说明

**DFP相关参数**：
- `--num_dfp`：动态特征池数量（默认8）
- `--dfp_start_iter`：开始构建DFP的迭代数（默认2000）
- `--selector_train_iter`：Selector训练轮数（默认50）
- `--dfp_reconstruct_interval`：DFP重构间隔（默认1000）
- `--max_global_features`：全局特征池最大容量（默认50000）

**网络参数**：
- `--embedding_dim`：特征嵌入维度（默认64）
- `--lambda_hcc`：HCC损失权重（默认0.1）

### 3. 监控训练过程

使用 wandb 监控训练指标：
- **阶段指示器**：`stage` (1/2/3)
- **DFP统计**：`dfp/global_pool_size`, `dfp/mean_dfp_size`
- **Selector性能**：`selector/loss`, `selector/accuracy`
- **训练损失**：`train/loss`, `train/loss_s`, `train/loss_c`, `train/loss_hcc`

## 数学原理

### 协方差矩阵计算
```
μ = (1/N) Σ f_i                    # 均值向量
Σ = (1/(N-1)) (F_global - μ1^T)(F_global - μ1^T)^T  # 协方差矩阵
```

### 相关性矩阵
```
R = A^(-1) Σ A^(-1)               # 标准化协方差矩阵
A = diag(σ_1, ..., σ_D)          # 标准差对角矩阵
```

### 特征分解与DFP分配
```
R = Q Λ Q^T                       # 特征值分解
P = [p_1, p_2, ..., p_num_dfp]   # 前num_dfp个特征向量
j* = argmax_j(|f_i^T p_j|)       # 特征分配规则
```

### Selector目标生成
```
j_target = argmin_j(||f_region - μ_j||_2)  # 最近DFP中心
```

## 实验配置建议

### 小规模实验（快速验证）
```bash
--num_dfp 4 \
--dfp_start_iter 1000 \
--selector_train_iter 25 \
--max_iteration 5000
```

### 中等规模实验（标准配置）
```bash
--num_dfp 8 \
--dfp_start_iter 2000 \
--selector_train_iter 50 \
--max_iteration 15000
```

### 大规模实验（完整性能）
```bash
--num_dfp 16 \
--dfp_start_iter 3000 \
--selector_train_iter 100 \
--max_iteration 20000
```

## 故障排除

### 常见问题

1. **DFP构建失败**
   - 检查全局特征池大小是否足够
   - 确保 `dfp_start_iter` 设置合理
   - 验证特征维度一致性

2. **Selector训练不收敛**
   - 调整学习率和训练轮数
   - 检查DFP分布是否均衡
   - 验证目标标签生成逻辑

3. **内存不足**
   - 减少 `max_global_features`
   - 调整批处理大小
   - 使用梯度累积

### 调试技巧

启用详细日志：
```python
logging.getLogger().setLevel(logging.DEBUG)
```

检查DFP统计信息：
```python
stats = cov_dfp.get_statistics()
print(f"DFP统计: {stats}")
```

## 扩展和定制

### 添加新的Selector架构
在 `networks/selector_network.py` 中实现新的选择器类，并在 `create_selector` 函数中注册。

### 自定义DFP构建策略
继承 `CovarianceDynamicFeaturePool` 类并重写 `build_dfps` 方法。

### 集成其他损失函数
在训练脚本中添加额外的损失项，并相应调整优化器。

## 性能期望

基于初步实验，该框架预期能够：
- 提高特征表示的专业化程度
- 增强模型对不同结构模式的适应性
- 在半监督学习任务中获得更好的性能
- 提供更稳定的训练过程

## 引用

如果使用本框架，请引用相关的研究论文和技术报告。 