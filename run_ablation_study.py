#!/usr/bin/env python3
"""
消融实验脚本：集成结构对齐与度量学习的动态特征池（Cov-DFP）框架
比较不同度量学习参数配置的效果
"""

import os
import subprocess
import sys
import time
from datetime import datetime


def run_experiment(config_name, args_dict, base_args):
    """运行单个实验配置"""
    print(f"\n{'='*60}")
    print(f"开始实验: {config_name}")
    print(f"{'='*60}")
    
    # 构建命令参数
    cmd = ["python", "train_cov_dfp_3d.py"] + base_args
    
    # 添加实验特定参数
    for key, value in args_dict.items():
        cmd.extend([f"--{key}", str(value)])
    
    # 显示完整命令
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行实验
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ 实验 {config_name} 完成！耗时: {duration/3600:.2f} 小时")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验 {config_name} 失败！错误码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ 实验 {config_name} 被用户中断！")
        return False


def main():
    """主函数：定义并执行消融实验"""
    print("=== 集成结构对齐与度量学习的动态特征池（Cov-DFP）消融实验 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 基础参数（所有实验共享）
    base_args = [
        "--dataset_name", "LA",
        "--dataset_path", "/home/jovyan/work/medical_dataset/LA",  # 根据实际路径修改
        "--model", "corn",
        "--gpu", "0",
        "--max_iteration", "15000",
        "--labeled_bs", "2",
        "--batch_size", "4",
        "--base_lr", "0.01",
        "--labelnum", "4",
        "--seed", "1337",
        "--lamda", "0.5",
        "--consistency", "1.0",
        "--consistency_rampup", "40.0",
        "--lambda_hcc", "0.1",
        "--use_dfp",
        "--num_dfp", "8",
        "--dfp_start_iter", "2000",
        "--selector_train_iter", "50",
        "--dfp_reconstruct_interval", "1000",
        "--max_global_features", "50000",
        "--embedding_dim", "64",
        "--hcc_weights", "0.5,0.5,1,1,1.5",
        "--cov_mode", "patch",
        "--patch_size", "4",
        "--hcc_patch_strategy", "mean_cov",
        "--hcc_metric", "fro",
        "--hcc_scale", "1.0",
        "--use_wandb",
        "--wandb_project", "Cov-DFP-Ablation"
    ]
    
    # 实验配置定义
    experiments = [
        # 1. 基线：无度量学习损失
        {
            "name": "baseline_no_metric_learning",
            "args": {
                "exp": "baseline_no_metric_learning",
                "lambda_compact": 0.0,
                "lambda_separate": 0.0,
                "separation_margin": 1.0
            }
        },
        
        # 2. 仅池内紧凑性损失
        {
            "name": "compact_only",
            "args": {
                "exp": "compact_only",
                "lambda_compact": 0.1,
                "lambda_separate": 0.0,
                "separation_margin": 1.0
            }
        },
        
        # 3. 仅池间分离性损失
        {
            "name": "separate_only", 
            "args": {
                "exp": "separate_only",
                "lambda_compact": 0.0,
                "lambda_separate": 0.05,
                "separation_margin": 1.0
            }
        },
        
        # 4. 完整度量学习框架（保守设置）
        {
            "name": "metric_learning_conservative",
            "args": {
                "exp": "metric_learning_conservative",
                "lambda_compact": 0.05,
                "lambda_separate": 0.02,
                "separation_margin": 1.0
            }
        },
        
        # 5. 完整度量学习框架（平衡设置）
        {
            "name": "metric_learning_balanced",
            "args": {
                "exp": "metric_learning_balanced",
                "lambda_compact": 0.1,
                "lambda_separate": 0.05,
                "separation_margin": 1.0
            }
        },
        
        # 6. 完整度量学习框架（激进设置）
        {
            "name": "metric_learning_aggressive",
            "args": {
                "exp": "metric_learning_aggressive",
                "lambda_compact": 0.15,
                "lambda_separate": 0.08,
                "separation_margin": 1.5
            }
        },
        
        # 7. 不同边际参数测试
        {
            "name": "margin_0.5",
            "args": {
                "exp": "margin_0.5",
                "lambda_compact": 0.1,
                "lambda_separate": 0.05,
                "separation_margin": 0.5
            }
        },
        
        {
            "name": "margin_2.0",
            "args": {
                "exp": "margin_2.0", 
                "lambda_compact": 0.1,
                "lambda_separate": 0.05,
                "separation_margin": 2.0
            }
        },
        
        # 8. 不同DFP数量测试
        {
            "name": "dfp_4_metric_learning",
            "args": {
                "exp": "dfp_4_metric_learning",
                "num_dfp": "4",
                "lambda_compact": 0.1,
                "lambda_separate": 0.05,
                "separation_margin": 1.0
            }
        },
        
        {
            "name": "dfp_12_metric_learning",
            "args": {
                "exp": "dfp_12_metric_learning", 
                "num_dfp": "12",
                "lambda_compact": 0.1,
                "lambda_separate": 0.05,
                "separation_margin": 1.0
            }
        }
    ]
    
    # 执行实验
    total_experiments = len(experiments)
    successful_experiments = 0
    failed_experiments = []
    
    print(f"\n总共需要运行 {total_experiments} 个实验\n")
    
    for i, exp_config in enumerate(experiments, 1):
        config_name = exp_config["name"]
        args_dict = exp_config["args"]
        
        print(f"进度: {i}/{total_experiments}")
        
        success = run_experiment(config_name, args_dict, base_args)
        
        if success:
            successful_experiments += 1
        else:
            failed_experiments.append(config_name)
        
        # 实验间休息片刻
        if i < total_experiments:
            print(f"\n休息10秒后开始下一个实验...\n")
            time.sleep(10)
    
    # 实验总结
    print(f"\n{'='*80}")
    print("实验总结")
    print(f"{'='*80}")
    print(f"总实验数: {total_experiments}")
    print(f"成功实验数: {successful_experiments}")
    print(f"失败实验数: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"\n失败的实验:")
        for exp_name in failed_experiments:
            print(f"  - {exp_name}")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("所有实验完成！")


def run_quick_test():
    """快速测试：运行短时间的实验来验证配置"""
    print("=== 快速测试模式 ===")
    print("运行短时间实验来验证配置...")
    
    base_args = [
        "--dataset_name", "LA",
        "--dataset_path", "/home/jovyan/work/medical_dataset/LA",
        "--model", "corn",
        "--gpu", "0",
        "--max_iteration", "100",  # 短时间测试
        "--labeled_bs", "2",
        "--batch_size", "4",
        "--base_lr", "0.01",
        "--labelnum", "4",
        "--seed", "1337",
        "--use_dfp",
        "--num_dfp", "4",
        "--dfp_start_iter", "50",
        "--selector_train_iter", "10",
        "--embedding_dim", "32",
        "--use_wandb",
        "--wandb_project", "Cov-DFP-QuickTest"
    ]
    
    test_config = {
        "exp": "quick_test_metric_learning",
        "lambda_compact": 0.1,
        "lambda_separate": 0.05,
        "separation_margin": 1.0
    }
    
    success = run_experiment("quick_test", test_config, base_args)
    
    if success:
        print("\n✅ 快速测试成功！配置正确，可以运行完整实验。")
    else:
        print("\n❌ 快速测试失败！请检查配置。")
    
    return success


if __name__ == "__main__":
    print("集成结构对齐与度量学习的动态特征池（Cov-DFP）消融实验脚本")
    print("选择运行模式:")
    print("1. 完整消融实验")
    print("2. 快速测试")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == "1":
            print("\n启动完整消融实验...")
            main()
            break
        elif choice == "2":
            print("\n启动快速测试...")
            run_quick_test()
            break
        elif choice == "3":
            print("退出程序。")
            sys.exit(0)
        else:
            print("无效选择，请输入 1、2 或 3。") 