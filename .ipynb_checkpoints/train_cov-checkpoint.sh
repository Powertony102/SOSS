#!/bin/bash

# 激活你的 Python 虚拟环境（如果有的话）
# source /path/to/venv/bin/activate

# 运行训练脚本
python3 train_cov_dfp_3d.py \
  --dataset_name LA \
  --root_path ./ \
  --exp cov_dfp \
  --model corn \
  --max_iteration 12000 \
  --labeled_bs 2 \
  --batch_size 4 \
  --base_lr 0.01 \
  --labelnum 4 \
  --seed 1337 \
  --gpu 0 \
  --use_dfp \
  --num_dfp 8 \
  # --use_wandb \
  --wandb_project SOSS \
  --dfp_start_iter 2000 \
  --selector_train_iter 500 \
  --dfp_reconstruct_interval 3000 \
  --embedding_dim 128 \

# 你可以根据需要添加/修改参数