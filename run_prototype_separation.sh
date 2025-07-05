#!/bin/bash

# 使用原型分离功能的训练脚本示例
# 基于您的文档描述实现的类间分离模块

# 基础配置
EXP_NAME="corn_prototype_separation"
DATASET_PATH="/home/jovyan/work/medical_dataset/LA"
LABELNUM=4
SEED=1337
GPU=0

# 原型分离参数
USE_PROTOTYPE=true
PROTOTYPE_CONFIDENCE_THRESHOLD=0.8
PROTOTYPE_K=10
PROTOTYPE_UPDATE_MOMENTUM=0.9
PROTOTYPE_INTRA_WEIGHT=1.0
PROTOTYPE_INTER_WEIGHT=0.1
PROTOTYPE_MARGIN=1.0

# 训练参数
MAX_ITERATION=15000
BATCH_SIZE=4
LABELED_BS=2
BASE_LR=0.01
CONSISTENCY=1.0
CONSISTENCY_RAMPUP=40.0
LAMDA=0.5

# DFP参数（可选）
USE_DFP=true
EMBEDDING_DIM=64
MEMORY_NUM=256
NUM_EIGENVECTORS=8

echo "开始训练带有原型分离功能的模型..."
echo "实验名称: $EXP_NAME"
echo "原型分离参数:"
echo "  - 置信度阈值: $PROTOTYPE_CONFIDENCE_THRESHOLD"
echo "  - 每类原型候选数: $PROTOTYPE_K"
echo "  - 类内紧致权重: $PROTOTYPE_INTRA_WEIGHT"
echo "  - 类间分离权重: $PROTOTYPE_INTER_WEIGHT"
echo "  - 分离边界: $PROTOTYPE_MARGIN"

# 仅使用原型分离（不使用DFP）
python train_cov_dfp_3d.py \
    --exp $EXP_NAME \
    --dataset_name LA \
    --dataset_path $DATASET_PATH \
    --model corn \
    --labelnum $LABELNUM \
    --seed $SEED \
    --gpu $GPU \
    --max_iteration $MAX_ITERATION \
    --batch_size $BATCH_SIZE \
    --labeled_bs $LABELED_BS \
    --base_lr $BASE_LR \
    --lamda $LAMDA \
    --consistency $CONSISTENCY \
    --consistency_rampup $CONSISTENCY_RAMPUP \
    --use_prototype \
    --prototype_confidence_threshold $PROTOTYPE_CONFIDENCE_THRESHOLD \
    --prototype_k $PROTOTYPE_K \
    --prototype_update_momentum $PROTOTYPE_UPDATE_MOMENTUM \
    --prototype_intra_weight $PROTOTYPE_INTRA_WEIGHT \
    --prototype_inter_weight $PROTOTYPE_INTER_WEIGHT \
    --prototype_margin $PROTOTYPE_MARGIN \
    --embedding_dim $EMBEDDING_DIM \
    --deterministic 1 \
    --use_wandb

echo "训练完成！"

# 使用原型分离 + DFP的组合训练
echo "开始训练带有原型分离和DFP的组合模型..."

EXP_NAME="corn_prototype_dfp_combined"

python train_cov_dfp_3d.py \
    --exp $EXP_NAME \
    --dataset_name LA \
    --dataset_path $DATASET_PATH \
    --model corn \
    --labelnum $LABELNUM \
    --seed $SEED \
    --gpu $GPU \
    --max_iteration $MAX_ITERATION \
    --batch_size $BATCH_SIZE \
    --labeled_bs $LABELED_BS \
    --base_lr $BASE_LR \
    --lamda $LAMDA \
    --consistency $CONSISTENCY \
    --consistency_rampup $CONSISTENCY_RAMPUP \
    --use_prototype \
    --prototype_confidence_threshold $PROTOTYPE_CONFIDENCE_THRESHOLD \
    --prototype_k $PROTOTYPE_K \
    --prototype_update_momentum $PROTOTYPE_UPDATE_MOMENTUM \
    --prototype_intra_weight $PROTOTYPE_INTRA_WEIGHT \
    --prototype_inter_weight $PROTOTYPE_INTER_WEIGHT \
    --prototype_margin $PROTOTYPE_MARGIN \
    --use_dfp \
    --embedding_dim $EMBEDDING_DIM \
    --memory_num $MEMORY_NUM \
    --num_eigenvectors $NUM_EIGENVECTORS \
    --max_store 10000 \
    --ema_alpha 0.9 \
    --consistency_o 0.05 \
    --num_filtered 12800 \
    --deterministic 1 \
    --use_wandb

echo "组合训练完成！" 