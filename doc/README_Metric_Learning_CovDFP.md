# 集成结构对齐与度量学习的动态特征池（Cov-DFP）框架

## 概述

本框架实现了论文研究方案《集成结构对齐与度量学习的动态特征池（Cov-DFP）框架》中提出的方法。该方案在原有Cov-DFP框架基础上，引入度量学习思想，通过**池内紧凑性损失**和**池间分离性损失**来优化特征空间结构，构建"界限分明、内部团结"的专家池系统。

## 核心思想

### 1. 池内紧凑性损失 (Intra-Pool Compactness Loss)
鼓励同一DFP内的特征相互靠近，形成紧密的、高内聚性的簇：

$$L_{\text{compact}_j}=\frac{1}{|B_j|}\sum_{f\in B_j}\|f - \mu_{B_j}\|_2^2$$

总紧凑性损失：
$$L_{\text{compact}}=\sum_{j\ \text{s.t.}\ B_j\neq\varnothing}L_{\text{compact}_j}$$

### 2. 池间分离性损失 (Inter-Pool Separation Loss)
将不同DFP的特征中心相互推远，确保各专家池在特征空间中拥有独立领域：

$$L_{\text{separate}}=\sum_{i = 1}^{\text{num\_dfp}}\sum_{j=i + 1}^{\text{num\_dfp}}\max(0,m-\|\mu_i-\mu_j\|_2^2)$$

### 3. 综合损失函数
$$L_{\text{total}}=L_{\text{main}}+\lambda_{\text{coral}}L_{\text{DFP\_CORAL}}+\lambda_{\text{compact}}L_{\text{compact}}+\lambda_{\text{separate}}L_{\text{separate}}$$

## 训练框架

### 阶段一：初始预训练
- 训练初步的主模型
- 建立全局特征池 F_global
- 不使用DFP，仅基础监督学习

### 阶段二：DFP构建
- 利用全局特征的二阶统计量构建结构化DFP
- 通过协方差分析提取主要共变模式
- 为Selector生成初始训练目标

### 阶段三：交替迭代训练
#### 模式A：Selector训练
- 冻结主模型，训练Selector网络
- 学习特征到DFP的映射关系

#### 模式B：主模型与DFP训练（核心改进）
- 使用Selector将特征分组到集合 {B₁, B₂, ..., B_num_dfp}
- 计算四项损失：
  - L_main：原始任务损失
  - L_DFP_CORAL：DFP协方差对齐损失
  - L_compact：池内紧凑性损失（新增）
  - L_separate：池间分离性损失（新增）

## 实现细节

### 新增文件和功能

#### 1. CovarianceDynamicFeaturePool增强
- `compute_intra_pool_compactness_loss()`: 计算池内紧凑性损失
- `compute_inter_pool_separation_loss()`: 计算池间分离性损失
- `compute_metric_learning_losses()`: 综合计算两个度量学习损失
- `group_features_by_dfp_predictions()`: 根据Selector预测分组特征

#### 2. 训练脚本更新
- 在`train_stage_three_main()`中集成度量学习损失
- 新增超参数：`lambda_compact`, `lambda_separate`, `separation_margin`
- 完善的wandb日志记录

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_compact` | 0.1 | 池内紧凑性损失权重 |
| `lambda_separate` | 0.05 | 池间分离性损失权重 |
| `separation_margin` | 1.0 | 池间分离边际参数 |
| `num_dfp` | 8 | 动态特征池数量 |
| `dfp_start_iter` | 2000 | 开始构建DFP的迭代数 |
| `selector_train_iter` | 50 | Selector训练迭代数 |
| `dfp_reconstruct_interval` | 1000 | DFP重构间隔 |

## 使用方法

### 1. 基本训练
```bash
python train_cov_dfp_3d.py \
    --use_dfp \
    --lambda_compact 0.1 \
    --lambda_separate 0.05 \
    --separation_margin 1.0 \
    --num_dfp 8 \
    --embedding_dim 64 \
    --use_wandb \
    --wandb_project "Cov-DFP-MetricLearning"
```

### 2. 使用提供的脚本
```bash
chmod +x run_cov_dfp_metric_learning.sh
./run_cov_dfp_metric_learning.sh
```

### 3. 测试实现
```bash
python test_metric_learning_losses.py
```

## GPU加速优化

框架全面支持GPU加速：
- 所有协方差计算在GPU上进行
- 特征池存储为GPU张量
- 度量学习损失计算GPU原生实现
- 自动设备检测和张量转移

## 超参数调优建议

### 度量学习权重
- `lambda_compact`: 建议范围 [0.05, 0.2]
  - 过小：池内特征分散，专家化不足
  - 过大：可能导致过度聚集，丧失多样性

- `lambda_separate`: 建议范围 [0.01, 0.1] 
  - 过小：池间区分度不足
  - 过大：可能导致训练不稳定

### 分离边际
- `separation_margin`: 建议范围 [0.5, 2.0]
  - 根据特征维度和池数量调整
  - 高维特征空间可适当增大

### 温度控制策略
可以实现动态权重调整：
```python
def get_lambda_compact(epoch, max_lambda=0.1):
    return max_lambda * ramps.sigmoid_rampup(epoch, 100)

def get_lambda_separate(epoch, max_lambda=0.05):
    return max_lambda * ramps.sigmoid_rampup(epoch, 150)
```

## 实验配置示例

### 配置1：保守设置（推荐初始尝试）
```bash
--lambda_compact 0.05 \
--lambda_separate 0.02 \
--separation_margin 1.0 \
--num_dfp 6
```

### 配置2：激进设置（追求强分离性）
```bash
--lambda_compact 0.15 \
--lambda_separate 0.08 \
--separation_margin 1.5 \
--num_dfp 10
```

### 配置3：平衡设置（论文默认）
```bash
--lambda_compact 0.1 \
--lambda_separate 0.05 \
--separation_margin 1.0 \
--num_dfp 8
```

## 监控指标

### Wandb日志内容
- `train/loss_compact`: 池内紧凑性损失
- `train/loss_separate`: 池间分离性损失
- `train/lambda_compact`: 紧凑性损失权重
- `train/lambda_separate`: 分离性损失权重
- `dfp/mean_dfp_size`: 平均DFP大小
- `selector/accuracy`: Selector预测准确率

### 关键观察指标
1. **损失平衡性**: 各损失项应处于同一数量级
2. **Selector准确率**: 应逐渐提升并稳定在较高水平
3. **DFP大小分布**: 各池大小应相对均衡
4. **训练稳定性**: 损失应平滑下降，无剧烈震荡

## 故障排除

### 常见问题

#### 1. 度量学习损失过大
**现象**: loss_compact或loss_separate数值异常大
**解决**: 
- 降低对应的lambda权重
- 检查特征归一化
- 调整separation_margin

#### 2. Selector训练不收敛
**现象**: Selector准确率持续低迷
**解决**:
- 增加selector_train_iter
- 调整Selector学习率
- 检查DFP构建质量

#### 3. GPU内存不足
**现象**: CUDA out of memory
**解决**:
- 减少batch_size
- 降低max_global_features
- 减少num_dfp

#### 4. 训练不稳定
**现象**: 损失震荡，模型性能波动大
**解决**:
- 降低度量学习权重
- 使用温度控制函数
- 增加dfp_reconstruct_interval

## 理论优势

### 1. 层次化优化
从三个层次约束特征空间：
- 单池内部结构（DFP-CORAL）
- 单池内聚性（紧凑性损失）
- 池间关系（分离性损失）

### 2. 高判别力特征
通过明确推远不同类别特征和拉近相同类别特征，学习更易分割分类的特征空间。

### 3. 鲁棒的DFP系统
DFP在训练过程中动态维持"专家"特性，防止概念漂移和模式退化。

## 引用

如果使用本框架，请引用相关论文：

```bibtex
@article{cov_dfp_metric_learning_2024,
  title={集成结构对齐与度量学习的动态特征池（Cov-DFP）框架},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 许可证

本项目遵循MIT许可证。详情请见LICENSE文件。 