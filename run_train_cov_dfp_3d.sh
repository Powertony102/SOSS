#!/bin/bash

# 运行 SOSS 训练脚本
# 注意：请根据你的实际环境修改数据集路径

# 激活虚拟环境（如果有的话）
# source /path/to/your/venv/bin/activate

# 设置数据集路径（请修改为你的实际路径）
DATASET_PATH="/home/jovyan/work/medical_dataset/LA"

# 运行训练
python3 train_cov_dfp_3d.py \
  --dataset_name LA \
  --root_path ./ \
  --dataset_path $DATASET_PATH \
  --exp cov_dfp_debug \
  --model corn \
  --max_iteration 15000 \
  --labeled_bs 2 \
  --batch_size 4 \
  --base_lr 0.01 \
  --labelnum 4 \
  --seed 1337 \
  --gpu 0 \
  --use_dfp \
  --num_dfp 8 \
  --dfp_start_iter 1000 \
  --selector_train_iter 250 \
  --dfp_reconstruct_interval 2000 \
  --max_global_features 50000 \
  --embedding_dim 128 \
  --use_wandb \
  --wandb_project SOSS \
  --consistency 1.0 \
  --consistency_rampup 40.0 \
  --lambda_hcc 0.1 \
  --hcc_weights "0.5,0.5,1,1,1.5" \
  --cov_mode patch \
  --patch_size 4 \
  --temperature 0.4 \
  --lamda 0.5 \
  --consistency_o 0.05