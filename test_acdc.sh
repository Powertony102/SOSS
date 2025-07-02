#!/bin/bash

# ACDC数据集测试脚本
# 用于测试训练好的ACDC心脏分割模型

# 默认参数
DATASET_NAME="ACDC"
ROOT_PATH="./"
DATASET_PATH="/home/jovyan/work/medical_dataset/ACDC_processed"
EXP="acdc_soss"
MODEL="corn2d"
MAX_ITERATION=20000
GPU="0"
DETAIL=1
LABELNUM=7
SEED=1337

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --exp)
            EXP="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max_iteration)
            MAX_ITERATION="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --labelnum)
            LABELNUM="$2"
            shift 2
            ;;
        --detail)
            DETAIL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --dataset_path PATH    ACDC预处理数据路径 (默认: $DATASET_PATH)"
            echo "  --exp NAME            实验名称 (默认: $EXP)"
            echo "  --model NAME          模型名称 (默认: $MODEL)"
            echo "  --max_iteration NUM   最大迭代次数 (默认: $MAX_ITERATION)"
            echo "  --gpu ID              GPU ID (默认: $GPU)"
            echo "  --labelnum NUM        标注数据数量 (默认: $LABELNUM)"
            echo "  --detail 0/1          是否显示详细结果 (默认: $DETAIL)"
            echo "  --seed NUM            随机种子 (默认: $SEED)"
            echo "  --help               显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "=== ACDC模型测试 ==="
echo "数据集路径: $DATASET_PATH"
echo "实验名称: $EXP"
echo "模型: $MODEL"
echo "最大迭代: $MAX_ITERATION"
echo "GPU: $GPU"
echo "标注数量: $LABELNUM"
echo "详细输出: $DETAIL"
echo "随机种子: $SEED"
echo "========================"

# 检查数据集路径
if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 数据集路径不存在: $DATASET_PATH"
    echo "请确保已经运行了ACDC数据预处理脚本"
    exit 1
fi

# 检查验证集文件
VAL_FILE="$DATASET_PATH/val.h5"
if [ ! -f "$VAL_FILE" ]; then
    echo "错误: 验证集文件不存在: $VAL_FILE"
    echo "请确保已经运行了ACDC数据预处理脚本"
    exit 1
fi

# 运行测试
python test_acdc.py \
    --dataset_name "$DATASET_NAME" \
    --root_path "$ROOT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --exp "$EXP" \
    --model "$MODEL" \
    --max_iteration "$MAX_ITERATION" \
    --gpu "$GPU" \
    --detail "$DETAIL" \
    --labelnum "$LABELNUM" \
    --seed "$SEED" \
    --num_dfp 8 \
    --lambda_compact 0.1 \
    --lambda_separate 0.05 \
    --memory_num 256 \
    --embedding_dim 64 \
    --num_filtered 12800 \
    --consistency 1.0 \
    --consistency_o 0.05 \
    --consistency_rampup 40.0

echo "测试完成!" 