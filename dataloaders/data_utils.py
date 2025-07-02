#!/usr/bin/env python3
"""
ACDC数据集的数据处理工具
用于处理数据形状、格式检查和修复
"""

import torch
import numpy as np
import logging

def check_and_fix_data_shape(volume_batch, label_batch=None, verbose=False):
    """
    检查并修复ACDC数据的形状问题
    
    Args:
        volume_batch: 输入图像批次，可能的形状：
                     - [B, C, H, W]: 标准4D格式
                     - [B, H, W]: 缺失通道维度
                     - [B, 4, H, W]: 错误的4通道输入
        label_batch: 标签批次 (可选)
        verbose: 是否打印详细信息
        
    Returns:
        tuple: (修复后的volume_batch, 修复后的label_batch)
    """
    if verbose:
        print(f"原始数据形状 - volume: {volume_batch.shape}", 
              f", label: {label_batch.shape if label_batch is not None else 'None'}")
    
    # 1. 修复图像数据
    original_shape = volume_batch.shape
    
    if len(volume_batch.shape) == 3:  # [B, H, W] -> [B, 1, H, W]
        volume_batch = volume_batch.unsqueeze(1)
        if verbose:
            print(f"修复: 添加通道维度 {original_shape} -> {volume_batch.shape}")
            
    elif len(volume_batch.shape) == 4:
        if volume_batch.shape[1] == 4:  # [B, 4, H, W] -> [B, 1, H, W]
            if verbose:
                print("检测到4通道输入，可能原因:")
                print("  - 数据预处理错误")
                print("  - 标签和图像数据混淆")
                print("  - One-hot编码被误当作输入通道")
                print("修复: 只使用第一个通道")
            volume_batch = volume_batch[:, 0:1, :, :]
            
        elif volume_batch.shape[1] != 1:
            if verbose:
                print(f"通道数异常: {volume_batch.shape[1]} (期望1)")
                print("修复: 只使用第一个通道")
            volume_batch = volume_batch[:, 0:1, :, :]
            
    elif len(volume_batch.shape) == 5:  # 可能是错误的3D格式
        if verbose:
            print("检测到5D输入，可能是错误的3D格式")
            print("修复: 去除深度维度")
        if volume_batch.shape[2] == 1:  # [B, C, 1, H, W] -> [B, C, H, W]
            volume_batch = volume_batch.squeeze(2)
        else:
            # 如果深度维度>1，选择中间切片
            depth_idx = volume_batch.shape[2] // 2
            volume_batch = volume_batch[:, :, depth_idx, :, :]
            if verbose:
                print(f"选择中间切片 (index {depth_idx})")
    
    # 2. 修复标签数据
    if label_batch is not None:
        if len(label_batch.shape) == 3:  # [B, H, W] - 正确格式
            pass
        elif len(label_batch.shape) == 4:  # [B, 1, H, W] -> [B, H, W]
            if label_batch.shape[1] == 1:
                label_batch = label_batch.squeeze(1)
                if verbose:
                    print("修复: 移除标签的多余通道维度")
            elif label_batch.shape[1] == 4:  # One-hot格式 -> 类别索引
                label_batch = torch.argmax(label_batch, dim=1)
                if verbose:
                    print("修复: 将One-hot标签转换为类别索引")
    
    # 3. 验证最终格式
    assert len(volume_batch.shape) == 4, f"修复后图像格式错误: {volume_batch.shape}"
    assert volume_batch.shape[1] == 1, f"修复后通道数错误: {volume_batch.shape[1]}"
    
    if label_batch is not None:
        assert len(label_batch.shape) == 3, f"修复后标签格式错误: {label_batch.shape}"
        assert volume_batch.shape[0] == label_batch.shape[0], "批次大小不匹配"
        assert volume_batch.shape[2:] == label_batch.shape[1:], "空间维度不匹配"
    
    if verbose:
        print(f"修复完成 - volume: {volume_batch.shape}", 
              f", label: {label_batch.shape if label_batch is not None else 'None'}")
    
    return volume_batch, label_batch


def validate_acdc_data(volume_batch, label_batch, expected_classes=4):
    """
    验证ACDC数据的有效性
    
    Args:
        volume_batch: 图像批次 [B, 1, H, W]
        label_batch: 标签批次 [B, H, W]
        expected_classes: 期望的类别数 (默认4: 背景+LV+RV+MYO)
        
    Returns:
        dict: 验证结果
    """
    result = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # 检查图像数据
    if volume_batch.dtype != torch.float32:
        result['warnings'].append(f"图像数据类型不是float32: {volume_batch.dtype}")
    
    if volume_batch.min() < 0 or volume_batch.max() > 1:
        result['warnings'].append(f"图像数值范围异常: [{volume_batch.min():.4f}, {volume_batch.max():.4f}]")
    
    # 检查标签数据
    if label_batch is not None:
        unique_labels = torch.unique(label_batch)
        if len(unique_labels) > expected_classes:
            result['errors'].append(f"标签类别数超出期望: {len(unique_labels)} > {expected_classes}")
            result['valid'] = False
        
        if torch.max(unique_labels) >= expected_classes:
            result['errors'].append(f"标签值超出范围: max={torch.max(unique_labels)} >= {expected_classes}")
            result['valid'] = False
        
        if torch.min(unique_labels) < 0:
            result['errors'].append(f"标签值包含负数: min={torch.min(unique_labels)}")
            result['valid'] = False
    
    return result


def diagnose_data_issue(volume_batch, label_batch=None):
    """
    诊断常见的ACDC数据问题
    
    Args:
        volume_batch: 原始图像数据
        label_batch: 原始标签数据
        
    Returns:
        dict: 诊断结果和建议
    """
    diagnosis = {
        'issues': [],
        'suggestions': []
    }
    
    # 诊断图像问题
    if len(volume_batch.shape) == 4 and volume_batch.shape[1] == 4:
        diagnosis['issues'].append("检测到4通道输入")
        diagnosis['suggestions'].extend([
            "可能原因1: 数据预处理时将4个类别的one-hot编码误作为输入通道",
            "可能原因2: 标签数据和图像数据位置颠倒",
            "可能原因3: 多帧数据被错误拼接",
            "建议: 检查数据加载和预处理流程"
        ])
    
    if len(volume_batch.shape) == 5:
        diagnosis['issues'].append("检测到5D输入")
        diagnosis['suggestions'].extend([
            "可能原因: 3D数据处理逻辑被错误应用到2D数据",
            "建议: 确认使用2D模型处理2D数据"
        ])
    
    # 诊断标签问题
    if label_batch is not None:
        if len(label_batch.shape) == 4 and label_batch.shape[1] == 4:
            diagnosis['issues'].append("检测到4通道标签")
            diagnosis['suggestions'].append("建议: 将one-hot标签转换为类别索引")
    
    if not diagnosis['issues']:
        diagnosis['issues'].append("未检测到明显问题")
        diagnosis['suggestions'].append("数据格式看起来正常")
    
    return diagnosis


class ACDCDataProcessor:
    """ACDC数据处理器类"""
    
    def __init__(self, expected_classes=4, verbose=False):
        self.expected_classes = expected_classes
        self.verbose = verbose
        self.stats = {
            'processed_batches': 0,
            'shape_fixes': 0,
            'channel_fixes': 0,
            'label_fixes': 0
        }
    
    def process_batch(self, volume_batch, label_batch=None):
        """处理单个批次的数据"""
        # 诊断问题
        if self.verbose:
            diagnosis = diagnose_data_issue(volume_batch, label_batch)
            if diagnosis['issues']:
                print("数据诊断:")
                for issue in diagnosis['issues']:
                    print(f"  问题: {issue}")
                for suggestion in diagnosis['suggestions']:
                    print(f"  建议: {suggestion}")
        
        # 修复数据
        original_volume_shape = volume_batch.shape
        volume_batch, label_batch = check_and_fix_data_shape(
            volume_batch, label_batch, verbose=self.verbose
        )
        
        # 验证数据
        validation = validate_acdc_data(volume_batch, label_batch, self.expected_classes)
        if not validation['valid']:
            for error in validation['errors']:
                logging.error(f"数据验证失败: {error}")
        
        for warning in validation['warnings']:
            logging.warning(f"数据警告: {warning}")
        
        # 更新统计
        self.stats['processed_batches'] += 1
        if original_volume_shape != volume_batch.shape:
            self.stats['shape_fixes'] += 1
        
        return volume_batch, label_batch, validation
    
    def get_stats(self):
        """获取处理统计信息"""
        return self.stats.copy()


def test_data_processor():
    """测试数据处理器"""
    print("测试ACDC数据处理器...")
    
    processor = ACDCDataProcessor(verbose=True)
    
    # 测试用例
    test_cases = [
        ("正常2D输入", torch.randn(2, 1, 256, 256), torch.randint(0, 4, (2, 256, 256))),
        ("4通道输入", torch.randn(2, 4, 256, 256), torch.randint(0, 4, (2, 256, 256))),
        ("缺失通道维度", torch.randn(2, 256, 256), torch.randint(0, 4, (2, 256, 256))),
        ("5D输入", torch.randn(2, 1, 1, 256, 256), torch.randint(0, 4, (2, 256, 256))),
    ]
    
    for name, volume, label in test_cases:
        print(f"\n=== 测试: {name} ===")
        try:
            volume_fixed, label_fixed, validation = processor.process_batch(volume, label)
            print(f"✓ 处理成功: {volume.shape} -> {volume_fixed.shape}")
            if not validation['valid']:
                print(f"⚠️ 验证警告: {validation['errors']}")
        except Exception as e:
            print(f"✗ 处理失败: {e}")
    
    print(f"\n统计信息: {processor.get_stats()}")


if __name__ == "__main__":
    test_data_processor() 