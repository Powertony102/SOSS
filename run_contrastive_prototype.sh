#!/bin/bash

# 运行基于SS-Net的对比学习原型分离训练
# ContrastivePrototypeManager 集成脚本

echo "🚀 开始基于SS-Net的对比学习原型分离训练..."

# 基础配置
DATASET_PATH="/home/jovyan/work/medical_dataset/LA"  # 修改为你的数据集路径
EXP_NAME="corn_contrastive_proto"
LABELNUM=4
MAX_ITER=15000

# # 1. 仅使用对比学习原型分离（不使用DFP）
# echo "📋 实验1: 仅使用对比学习原型分离（简单版本）"
# python train_cov_dfp_3d.py \
#     --dataset_path $DATASET_PATH \
#     --exp ${EXP_NAME}_simple \
#     --labelnum $LABELNUM \
#     --max_iteration $MAX_ITER \
#     --use_prototype \
#     --prototype_elements_per_class 32 \
#     --prototype_contrastive_weight 1.0 \
#     --prototype_intra_weight 0.1 \
#     --prototype_inter_weight 0.1 \
#     --prototype_confidence_threshold 0.8 \
#     --use_wandb \
#     --wandb_project "CORN-Contrastive-Prototype" \
#     --seed 1337

# echo "✅ 实验1完成"

# # 2. 对比学习原型分离 + DFP
# echo "📋 实验2: 对比学习原型分离 + DFP"
# python train_cov_dfp_3d.py \
#     --dataset_path $DATASET_PATH \
#     --exp ${EXP_NAME}_with_dfp \
#     --labelnum $LABELNUM \
#     --max_iteration $MAX_ITER \
#     --use_dfp \
#     --use_prototype \
#     --prototype_elements_per_class 32 \
#     --prototype_contrastive_weight 1.0 \
#     --prototype_intra_weight 0.1 \
#     --prototype_inter_weight 0.1 \
#     --prototype_confidence_threshold 0.8 \
#     --use_wandb \
#     --wandb_project "CORN-Contrastive-Prototype" \
#     --seed 1337

# echo "✅ 实验2完成"

# 3. 使用学习的选择器版本（高级）
echo "📋 实验3: 使用学习的选择器（高级版本）"
python train_cov_dfp_3d.py \
    --dataset_path $DATASET_PATH \
    --exp ${EXP_NAME}_learned_selector \
    --labelnum $LABELNUM \
    --max_iteration $MAX_ITER \
    --use_dfp \
    --use_prototype \
    --prototype_elements_per_class 32 \
    --prototype_contrastive_weight 0.1 \
    --prototype_intra_weight 0.1 \
    --prototype_inter_weight 0.1 \
    --prototype_confidence_threshold 0.8 \
    --prototype_use_learned_selector \
    --use_wandb \
    --wandb_project "new_SOSS" \
    --seed 1337 \
    --gpu 1

echo "✅ 实验3完成"

# # 4. 参数调优实验
# echo "📋 实验4: 参数调优 - 不同的对比学习权重"
# for weight in 0.5 1.0 2.0; do
#     echo "   测试对比学习权重: $weight"
#     python train_cov_dfp_3d.py \
#         --dataset_path $DATASET_PATH \
#         --exp ${EXP_NAME}_weight${weight} \
#         --labelnum $LABELNUM \
#         --max_iteration 5000 \
#         --use_dfp \
#         --use_prototype \
#         --prototype_elements_per_class 32 \
#         --prototype_contrastive_weight $weight \
#         --prototype_intra_weight 0.1 \
#         --prototype_inter_weight 0.1 \
#         --prototype_confidence_threshold 0.8 \
#         --use_wandb \
#         --wandb_project "CORN-Contrastive-Prototype-Ablation" \
#         --seed 1337
# done

# echo "✅ 实验4完成"

# # 5. 不同内存大小的消融实验
# echo "📋 实验5: 不同内存大小的消融实验"
# for elements in 16 32 64; do
#     echo "   测试每类元素数量: $elements"
#     python train_cov_dfp_3d.py \
#         --dataset_path $DATASET_PATH \
#         --exp ${EXP_NAME}_elements${elements} \
#         --labelnum $LABELNUM \
#         --max_iteration 5000 \
#         --use_dfp \
#         --use_prototype \
#         --prototype_elements_per_class $elements \
#         --prototype_contrastive_weight 1.0 \
#         --prototype_intra_weight 0.1 \
#         --prototype_inter_weight 0.1 \
#         --prototype_confidence_threshold 0.8 \
#         --use_wandb \
#         --wandb_project "CORN-Contrastive-Prototype-Ablation" \
#         --seed 1337
# done

echo "✅ 实验5完成"

echo "🎉 所有实验完成！"
echo ""
echo "📊 实验总结:"
echo "1. 实验1: 基础对比学习原型分离"
echo "2. 实验2: 对比学习原型分离 + DFP"
echo "3. 实验3: 使用学习的选择器"
echo "4. 实验4: 对比学习权重调优"
echo "5. 实验5: 内存大小消融实验"
echo ""
echo "💡 建议:"
echo "- 查看 wandb 项目 'CORN-Contrastive-Prototype' 对比结果"
echo "- 关注对比学习损失 (contrastive_loss) 的变化趋势"
echo "- 比较不同配置下的验证 Dice 分数"
echo "- 监控内存使用情况，根据GPU调整 elements_per_class" 