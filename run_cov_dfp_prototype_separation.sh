#!/bin/bash

# 集成结构对齐、度量学习与原型分离的动态特征池（Cov-DFP + Prototype Separation）框架
# 训练脚本

# 基础参数
DATASET_NAME="LA"
DATASET_PATH="/home/jovyan/work/medical_dataset/LA"  # 根据实际路径修改
EXP_NAME="cov_dfp_prototype_separation"
MODEL="corn"
GPU="1"

# 训练参数
MAX_ITERATION=15000
LABELED_BS=2
BATCH_SIZE=4
BASE_LR=0.01
LABELNUM=4
SEED=1337

# 损失函数权重
LAMDA=0.5
CONSISTENCY=1.0
CONSISTENCY_RAMPUP=40.0
LAMBDA_HCC=0.1

# DFP参数
NUM_DFP=4
DFP_START_ITER=2000
SELECTOR_TRAIN_ITER=150
DFP_RECONSTRUCT_INTERVAL=3000
MAX_GLOBAL_FEATURES=20000
EMBEDDING_DIM=128

# 度量学习参数
LAMBDA_COMPACT=0.05      # 池内紧凑性损失权重
LAMBDA_SEPARATE=0.02     # 池间分离性损失权重
SEPARATION_MARGIN=1.0    # 池间分离边际

# 原型分离参数（新增）
USE_PROTOTYPE_SEPARATION=true
LAMBDA_PROTOTYPE=0.3           # 原型分离损失权重
PROTO_MOMENTUM=0.95           # 原型更新动量
PROTO_CONF_THRESH=0.85        # 原型更新置信度阈值
PROTO_LAMBDA_INTRA=0.3        # 类内紧致性权重
PROTO_LAMBDA_INTER=0.1        # 类间分离权重
PROTO_MARGIN=1.5              # 类间分离边际
PROTO_UPDATE_INTERVAL=5       # 原型更新间隔（批次）

# HCC参数
HCC_WEIGHTS="0.5,0.5,1,1,1.5"
COV_MODE="patch"
PATCH_SIZE=4
HCC_PATCH_STRATEGY="mean_cov"
HCC_METRIC="fro"
HCC_SCALE=1.0

# Wandb配置
USE_WANDB=true
WANDB_PROJECT="SOSS"
WANDB_ENTITY=""  # 根据需要设置

echo "=== 开始训练集成原型分离的动态特征池（Cov-DFP + Prototype Separation）框架 ==="
echo "实验名称: $EXP_NAME"
echo "使用GPU: $GPU"
echo "DFP数量: $NUM_DFP"
echo "度量学习参数: λ_compact=$LAMBDA_COMPACT, λ_separate=$LAMBDA_SEPARATE, margin=$SEPARATION_MARGIN"
echo "原型分离参数: λ_proto=$LAMBDA_PROTOTYPE, momentum=$PROTO_MOMENTUM, conf_thresh=$PROTO_CONF_THRESH"

# 构建命令
CMD="python train_cov_dfp_3d.py \
    --dataset_name $DATASET_NAME \
    --dataset_path $DATASET_PATH \
    --exp $EXP_NAME \
    --model $MODEL \
    --gpu $GPU \
    --max_iteration $MAX_ITERATION \
    --labeled_bs $LABELED_BS \
    --batch_size $BATCH_SIZE \
    --base_lr $BASE_LR \
    --labelnum $LABELNUM \
    --seed $SEED \
    --lamda $LAMDA \
    --consistency $CONSISTENCY \
    --consistency_rampup $CONSISTENCY_RAMPUP \
    --lambda_hcc $LAMBDA_HCC \
    --use_dfp \
    --num_dfp $NUM_DFP \
    --dfp_start_iter $DFP_START_ITER \
    --selector_train_iter $SELECTOR_TRAIN_ITER \
    --dfp_reconstruct_interval $DFP_RECONSTRUCT_INTERVAL \
    --max_global_features $MAX_GLOBAL_FEATURES \
    --embedding_dim $EMBEDDING_DIM \
    --lambda_compact $LAMBDA_COMPACT \
    --lambda_separate $LAMBDA_SEPARATE \
    --separation_margin $SEPARATION_MARGIN \
    --hcc_weights $HCC_WEIGHTS \
    --cov_mode $COV_MODE \
    --patch_size $PATCH_SIZE \
    --hcc_patch_strategy $HCC_PATCH_STRATEGY \
    --hcc_metric $HCC_METRIC \
    --hcc_scale $HCC_SCALE"

# 添加原型分离参数
if [ "$USE_PROTOTYPE_SEPARATION" = true ]; then
    CMD="$CMD --use_prototype_separation \
        --lambda_prototype $LAMBDA_PROTOTYPE \
        --proto_momentum $PROTO_MOMENTUM \
        --proto_conf_thresh $PROTO_CONF_THRESH \
        --proto_lambda_intra $PROTO_LAMBDA_INTRA \
        --proto_lambda_inter $PROTO_LAMBDA_INTER \
        --proto_margin $PROTO_MARGIN \
        --proto_update_interval $PROTO_UPDATE_INTERVAL"
fi

# 添加wandb参数
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        CMD="$CMD --wandb_entity $WANDB_ENTITY"
    fi
fi

echo "执行命令:"
echo "$CMD"
echo ""

# 执行训练
eval $CMD

echo "=== 训练完成 ===" 