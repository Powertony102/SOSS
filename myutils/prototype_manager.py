import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class PrototypeManager:
    """
    原型管理器 - 实现类间分离模块
    设计理念：通过原型引导的特征聚类策略，将同类像素特征拉近、异类特征拉远
    """
    
    def __init__(self, num_classes: int, feature_dim: int, 
                 k_prototypes: int = 10, confidence_threshold: float = 0.8, 
                 update_momentum: float = 0.9, device: str = 'cuda'):
        """
        初始化原型管理器
        
        Args:
            num_classes: 类别数量
            feature_dim: 特征维度
            k_prototypes: 每类选择的候选原型数量
            confidence_threshold: 置信度阈值
            update_momentum: 滑动平均更新动量
            device: 设备
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.k_prototypes = k_prototypes
        self.confidence_threshold = confidence_threshold
        self.update_momentum = update_momentum
        self.device = device
        
        # 初始化原型存储 - 忽略背景类（类别0）
        self.prototypes = {}
        self.prototype_counts = {}
        self.initialized = False
        
        # 用于候选原型选择的缓存
        self.candidate_features = {c: [] for c in range(1, num_classes)}
        self.candidate_scores = {c: [] for c in range(1, num_classes)}
        
    def extract_high_confidence_features(self, features: torch.Tensor, 
                                       predictions: torch.Tensor, 
                                       labels: torch.Tensor = None,
                                       is_labeled: bool = True) -> Dict[int, torch.Tensor]:
        """
        提取高置信度特征
        
        Args:
            features: 特征张量 [N, C, H, W, D] 或 [N*H*W*D, C]
            predictions: 预测概率 [N, num_classes, H, W, D] 或 [N*H*W*D, num_classes]
            labels: 真实标签 [N, H, W, D] 或 [N*H*W*D] (仅标记数据)
            is_labeled: 是否为标记数据
            
        Returns:
            high_conf_features: 每类高置信度特征的字典
        """
        # 确保输入维度正确
        if features.dim() == 5:  # [N, C, H, W, D]
            N, C, H, W, D = features.shape
            features = features.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # [N*H*W*D, C]
        
        if predictions.dim() == 5:  # [N, num_classes, H, W, D]
            N, num_classes, H, W, D = predictions.shape
            predictions = predictions.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)  # [N*H*W*D, num_classes]
        
        if labels is not None and labels.dim() == 4:  # [N, H, W, D]
            labels = labels.view(-1)  # [N*H*W*D]
        
        # 计算预测概率和置信度
        probs = F.softmax(predictions, dim=1)  # [N*H*W*D, num_classes]
        max_probs, pred_classes = torch.max(probs, dim=1)  # [N*H*W*D]
        
        high_conf_features = {}
        
        # 对每个类别（忽略背景类）
        for class_id in range(1, self.num_classes):
            if is_labeled:
                # 对于标记数据，使用真实标签且要求高置信度
                if labels is not None:
                    class_mask = (labels == class_id)
                    conf_mask = (max_probs > self.confidence_threshold)
                    pred_correct_mask = (pred_classes == class_id)
                    
                    # 三个条件都满足：真实标签匹配 + 高置信度 + 预测正确
                    valid_mask = class_mask & conf_mask & pred_correct_mask
                else:
                    # 如果没有真实标签，只使用预测和置信度
                    pred_mask = (pred_classes == class_id)
                    conf_mask = (max_probs > self.confidence_threshold)
                    valid_mask = pred_mask & conf_mask
            else:
                # 对于无标记数据，使用伪标签且要求高置信度
                pred_mask = (pred_classes == class_id)
                conf_mask = (max_probs > self.confidence_threshold)
                valid_mask = pred_mask & conf_mask
            
            if valid_mask.sum() > 0:
                class_features = features[valid_mask]  # [N_valid, C]
                class_confidences = max_probs[valid_mask]  # [N_valid]
                
                # 按置信度或特征范数排序，选择前k个
                if len(class_features) > self.k_prototypes:
                    # 计算特征范数
                    feature_norms = torch.norm(class_features, dim=1)
                    # 综合考虑置信度和特征范数
                    combined_scores = class_confidences * feature_norms
                    _, top_indices = torch.topk(combined_scores, self.k_prototypes)
                    class_features = class_features[top_indices]
                
                high_conf_features[class_id] = class_features
        
        return high_conf_features
    
    def initialize_prototypes(self, features_dict: Dict[int, torch.Tensor]):
        """
        初始化原型
        
        Args:
            features_dict: 每类特征的字典
        """
        for class_id, features in features_dict.items():
            if class_id == 0:  # 忽略背景类
                continue
                
            if len(features) > 0:
                # 计算初始原型（平均值）
                prototype = torch.mean(features, dim=0)  # [feature_dim]
                self.prototypes[class_id] = prototype.to(self.device)
                self.prototype_counts[class_id] = len(features)
        
        self.initialized = True
    
    def update_prototypes(self, features_dict: Dict[int, torch.Tensor]):
        """
        在线更新原型
        
        Args:
            features_dict: 每类新特征的字典
        """
        if not self.initialized:
            self.initialize_prototypes(features_dict)
            return
        
        for class_id, features in features_dict.items():
            if class_id == 0:  # 忽略背景类
                continue
                
            if len(features) > 0:
                # 计算新特征的平均值
                new_prototype = torch.mean(features, dim=0)  # [feature_dim]
                
                if class_id in self.prototypes:
                    # 滑动平均更新
                    old_prototype = self.prototypes[class_id]
                    updated_prototype = (self.update_momentum * old_prototype + 
                                       (1 - self.update_momentum) * new_prototype)
                    self.prototypes[class_id] = updated_prototype.to(self.device)
                    self.prototype_counts[class_id] += len(features)
                else:
                    # 新类别，直接设置
                    self.prototypes[class_id] = new_prototype.to(self.device)
                    self.prototype_counts[class_id] = len(features)
    
    def get_prototypes(self) -> Dict[int, torch.Tensor]:
        """获取当前原型"""
        return self.prototypes
    
    def compute_intra_class_loss(self, features: torch.Tensor, 
                                predictions: torch.Tensor, 
                                labels: torch.Tensor = None,
                                is_labeled: bool = True) -> torch.Tensor:
        """
        计算类内紧致损失
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            
        Returns:
            intra_class_loss: 类内紧致损失
        """
        if not self.initialized or len(self.prototypes) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 提取高置信度特征
        high_conf_features = self.extract_high_confidence_features(
            features, predictions, labels, is_labeled
        )
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0
        
        for class_id, class_features in high_conf_features.items():
            if class_id in self.prototypes:
                prototype = self.prototypes[class_id]  # [feature_dim]
                
                # 计算特征到原型的距离
                distances = torch.norm(class_features - prototype.unsqueeze(0), dim=1)  # [N_class]
                class_loss = torch.mean(distances ** 2)  # L2距离的平方
                
                total_loss += class_loss
                num_valid += 1
        
        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_inter_class_loss(self, margin: float = 1.0) -> torch.Tensor:
        """
        计算类间分离损失
        
        Args:
            margin: 最小分离距离
            
        Returns:
            inter_class_loss: 类间分离损失
        """
        if not self.initialized or len(self.prototypes) < 2:
            return torch.tensor(0.0, device=self.device)
        
        prototype_list = list(self.prototypes.values())
        class_ids = list(self.prototypes.keys())
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        
        # 计算所有类别对之间的分离损失
        for i in range(len(prototype_list)):
            for j in range(i + 1, len(prototype_list)):
                prototype_i = prototype_list[i]
                prototype_j = prototype_list[j]
                
                # 计算欧氏距离
                distance = torch.norm(prototype_i - prototype_j)
                
                # 分离损失：如果距离小于margin，则施加惩罚
                separation_loss = torch.max(torch.tensor(0.0, device=self.device), 
                                          margin - distance) ** 2
                
                total_loss += separation_loss
                num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_prototype_loss(self, features: torch.Tensor, 
                              predictions: torch.Tensor, 
                              labels: torch.Tensor = None,
                              is_labeled: bool = True,
                              intra_weight: float = 1.0,
                              inter_weight: float = 0.1,
                              margin: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算完整的原型损失
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            intra_weight: 类内紧致损失权重
            inter_weight: 类间分离损失权重
            margin: 类间分离的最小距离
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情
        """
        # 计算类内紧致损失
        intra_loss = self.compute_intra_class_loss(features, predictions, labels, is_labeled)
        
        # 计算类间分离损失
        inter_loss = self.compute_inter_class_loss(margin)
        
        # 总损失
        total_loss = intra_weight * intra_loss + inter_weight * inter_loss
        
        loss_dict = {
            'intra_loss': intra_loss.item(),
            'inter_loss': inter_loss.item(),
            'total_prototype_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def update_and_compute_loss(self, features: torch.Tensor, 
                               predictions: torch.Tensor, 
                               labels: torch.Tensor = None,
                               is_labeled: bool = True,
                               intra_weight: float = 1.0,
                               inter_weight: float = 0.1,
                               margin: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        更新原型并计算损失（一体化接口）
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            intra_weight: 类内紧致损失权重
            inter_weight: 类间分离损失权重
            margin: 类间分离的最小距离
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情
        """
        # 提取高置信度特征用于更新原型
        high_conf_features = self.extract_high_confidence_features(
            features, predictions, labels, is_labeled
        )
        
        # 更新原型
        if len(high_conf_features) > 0:
            self.update_prototypes(high_conf_features)
        
        # 计算损失
        total_loss, loss_dict = self.compute_prototype_loss(
            features, predictions, labels, is_labeled, 
            intra_weight, inter_weight, margin
        )
        
        return total_loss, loss_dict
    
    def get_prototype_info(self) -> Dict[str, any]:
        """获取原型信息"""
        info = {
            'num_prototypes': len(self.prototypes),
            'prototype_counts': self.prototype_counts.copy(),
            'initialized': self.initialized
        }
        
        if self.initialized and len(self.prototypes) > 1:
            # 计算类间距离统计
            prototype_list = list(self.prototypes.values())
            distances = []
            for i in range(len(prototype_list)):
                for j in range(i + 1, len(prototype_list)):
                    dist = torch.norm(prototype_list[i] - prototype_list[j]).item()
                    distances.append(dist)
            
            if distances:
                info['inter_class_distances'] = {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': np.min(distances),
                    'max': np.max(distances)
                }
        
        return info 