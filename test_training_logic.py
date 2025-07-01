 #!/usr/bin/env python3
"""
测试训练逻辑的简化脚本
用于验证阶段转换是否正确
"""

def test_training_logic():
    # 模拟参数
    dfp_start_iter = 5
    selector_train_iter = 3
    dfp_reconstruct_interval = 10
    max_iterations = 25
    
    # 模拟状态变量
    dfps_built = False
    selector_trained = False
    selector_train_counter = 0
    last_dfp_reconstruct_iter = 0
    
    print("=== 训练逻辑测试 ===")
    print(f"dfp_start_iter: {dfp_start_iter}")
    print(f"selector_train_iter: {selector_train_iter}")
    print(f"dfp_reconstruct_interval: {dfp_reconstruct_interval}")
    print(f"max_iterations: {max_iterations}")
    print()
    
    for iter_num in range(1, max_iterations + 1):
        print(f"Iter {iter_num:2d}: ", end="")
        
        if iter_num < dfp_start_iter:
            print("Stage 1 - 初始预训练")
            
        elif iter_num == dfp_start_iter:
            print("Stage 2 - 构建DFP")
            dfps_built = True
            last_dfp_reconstruct_iter = iter_num
            selector_trained = False
            selector_train_counter = 0
            
        elif iter_num > dfp_start_iter and dfps_built:
            # 检查是否需要重构DFP
            if iter_num - last_dfp_reconstruct_iter >= dfp_reconstruct_interval:
                print("Stage 3 - 重构DFP", end=" -> ")
                last_dfp_reconstruct_iter = iter_num
                selector_trained = False
                selector_train_counter = 0
            
            # 训练逻辑
            if not selector_trained:
                if selector_train_counter < selector_train_iter:
                    print(f"Stage 3A - 训练Selector ({selector_train_counter+1}/{selector_train_iter})")
                    selector_train_counter += 1
                    
                    if selector_train_counter >= selector_train_iter:
                        selector_trained = True
                        print(f"         -> Selector训练完成!")
            else:
                print("Stage 3B - 训练主模型")
        
        print(f"         状态: dfps_built={dfps_built}, selector_trained={selector_trained}, selector_counter={selector_train_counter}")

if __name__ == "__main__":
    test_training_logic()