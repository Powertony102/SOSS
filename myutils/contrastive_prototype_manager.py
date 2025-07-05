import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class ContrastivePrototypeManager:
    """
    基于 SemiSeg-Contrastive 的对比学习原型管理器
    参考: https://github.com/Shathe/SemiSeg-Contrastive
    
    核心思想：
    1. 每类保留多个高质量特征向量作为内存
    2. 使用学习的选择器评估特征重要性
    3. 计算当前特征与内存特征的对比学习损失
    4. 基于相似性和距离的正确损失计算
    """
    
    def __init__(self, num_classes: int, feature_dim: int, 
                 elements_per_class: int = 32, confidence_threshold: float = 0.8,
                 use_learned_selector: bool = False, device: str = 'cuda'):
        """
        初始化对比学习原型管理器
        
        Args:
            num_classes: 类别数量
            feature_dim: 特征维度
            elements_per_class: 每类保留的特征数量
            confidence_threshold: 置信度阈值
            use_learned_selector: 是否使用学习的特征选择器
            device: 设备
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.elements_per_class = elements_per_class
        self.confidence_threshold = confidence_threshold
        self.use_learned_selector = use_learned_selector
        self.device = device
        
        # 特征内存 - 每个类别保留多个高质量特征
        self.memory = [None] * num_classes
        self.initialized = False
        
        # 学习的特征选择器（可选）
        if use_learned_selector:
            self.feature_selectors = nn.ModuleDict()
            self.memory_selectors = nn.ModuleDict()
            
            for c in range(num_classes):
                # 当前特征选择器
                self.feature_selectors[f'contrastive_class_selector_{c}'] = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 4),
                    nn.ReLU(),
                    nn.Linear(feature_dim // 4, 1)
                ).to(device)
                
                # 内存特征选择器
                self.memory_selectors[f'contrastive_class_selector_memory{c}'] = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 4),
                    nn.ReLU(),
                    nn.Linear(feature_dim // 4, 1)
                ).to(device)
        else:
            self.feature_selectors = None
            self.memory_selectors = None
    
    def extract_high_quality_features(self, features: torch.Tensor, 
                                    predictions: torch.Tensor, 
                                    labels: torch.Tensor = None,
                                    is_labeled: bool = True) -> Dict[int, torch.Tensor]:
        """
        提取高质量特征 - 按照SS-Net的方式
        
        Args:
            features: 特征张量 [N, C, H, W, D] 或 [N*H*W*D, C]
            predictions: 预测概率 [N, num_classes, H, W, D] 或 [N*H*W*D, num_classes]
            labels: 真实标签 [N, H, W, D] 或 [N*H*W*D] (仅标记数据)
            is_labeled: 是否为标记数据
            
        Returns:
            high_quality_features: 每类高质量特征的字典
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
        
        # 分离梯度用于内存更新
        features_detached = features.detach()
        predictions_detached = predictions.detach()
        if labels is not None:
            labels_detached = labels.detach()
        else:
            labels_detached = None
        
        # 计算预测概率和置信度
        probs = F.softmax(predictions_detached, dim=1)  # [N*H*W*D, num_classes]
        max_probs, pred_classes = torch.max(probs, dim=1)  # [N*H*W*D]
        
        high_quality_features = {}
        
        # 对每个类别（包括背景类，与SS-Net一致）
        for class_id in range(self.num_classes):
            if is_labeled:
                # 对于标记数据，使用真实标签
                if labels_detached is not None:
                    class_mask = (labels_detached == class_id)
                    conf_mask = (max_probs > self.confidence_threshold)
                    valid_mask = class_mask & conf_mask
                else:
                    # 如果没有真实标签，使用预测
                    pred_mask = (pred_classes == class_id)
                    conf_mask = (max_probs > self.confidence_threshold)
                    valid_mask = pred_mask & conf_mask
            else:
                # 对于无标记数据，使用伪标签
                pred_mask = (pred_classes == class_id)
                conf_mask = (max_probs > self.confidence_threshold)
                valid_mask = pred_mask & conf_mask
            
            if valid_mask.sum() > 0:
                class_features = features_detached[valid_mask]  # [N_valid, C]
                
                # 按照SS-Net的方式选择特征
                if len(class_features) > self.elements_per_class:
                    if self.use_learned_selector and self.feature_selectors is not None:
                        # 使用学习的选择器
                        selector_key = f'contrastive_class_selector_{class_id}'
                        if selector_key in self.feature_selectors:
                            with torch.no_grad():
                                scores = self.feature_selectors[selector_key](class_features)
                                scores = torch.sigmoid(scores)
                                _, top_indices = torch.topk(scores.squeeze(), self.elements_per_class)
                                selected_features = class_features[top_indices]
                        else:
                            # 后备方案：随机选择
                            indices = torch.randperm(len(class_features))[:self.elements_per_class]
                            selected_features = class_features[indices]
                    else:
                        # 简单方案：基于置信度选择
                        class_confidences = max_probs[valid_mask]
                        _, top_indices = torch.topk(class_confidences, self.elements_per_class)
                        selected_features = class_features[top_indices]
                else:
                    selected_features = class_features
                
                if len(selected_features) > 0:
                    high_quality_features[class_id] = selected_features
        
        return high_quality_features
    
    def update_memory(self, features_dict: Dict[int, torch.Tensor]):
        """
        更新特征内存 - 按照SS-Net的在线替换策略
        
        Args:
            features_dict: 每类新特征的字典
        """
        for class_id, new_features in features_dict.items():
            if len(new_features) > 0:
                # 直接替换内存（SS-Net的方式）
                self.memory[class_id] = new_features.detach().cpu().numpy()
        
        self.initialized = True
    
    def contrastive_class_to_class_learned_memory(self, features: torch.Tensor, 
                                                 class_labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失 - 基于SS-Net的实现
        
        Args:
            features: 特征张量 [N, feature_dim] (已经过投影头处理)
            class_labels: 类别标签 [N]
            
        Returns:
            contrastive_loss: 对比学习损失
        """
        if not self.initialized:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_valid_classes = 0
        
        # 对每个类别计算对比损失
        for c in range(self.num_classes):
            # 获取当前类别的特征
            mask_c = class_labels == c
            features_c = features[mask_c, :]  # [M, feature_dim]
            memory_c = self.memory[c]  # numpy array [N, feature_dim]
            
            # 检查是否有足够的特征进行对比学习
            if (memory_c is not None and 
                features_c.shape[0] > 1 and 
                memory_c.shape[0] > 1):
                
                # 转换内存特征为tensor
                memory_c = torch.from_numpy(memory_c).to(self.device)  # [N, feature_dim]
                
                # L2 归一化特征向量
                memory_c_norm = F.normalize(memory_c, dim=1)  # [N, feature_dim]
                features_c_norm = F.normalize(features_c, dim=1)  # [M, feature_dim]
                
                # 计算相似性矩阵
                similarities = torch.mm(features_c_norm, memory_c_norm.transpose(1, 0))  # [M, N]
                # 转换为距离 (值在[0,2]之间，0表示相同向量)
                distances = 1 - similarities  # [M, N]
                
                # 计算特征权重
                if self.use_learned_selector and self.feature_selectors is not None:
                    # 使用学习的选择器计算权重
                    selector_key = f'contrastive_class_selector_{c}'
                    memory_selector_key = f'contrastive_class_selector_memory{c}'
                    
                    if (selector_key in self.feature_selectors and 
                        memory_selector_key in self.memory_selectors):
                        
                        # 当前特征的权重
                        learned_weights_features = self.feature_selectors[selector_key](features_c.detach())
                        learned_weights_features = torch.sigmoid(learned_weights_features)  # [M, 1]
                        
                        # 重新缩放权重
                        rescaled_weights = (learned_weights_features.shape[0] / 
                                          learned_weights_features.sum(dim=0)) * learned_weights_features
                        rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])  # [M, N]
                        distances = distances * rescaled_weights
                        
                        # 内存特征的权重
                        learned_weights_memory = self.memory_selectors[memory_selector_key](memory_c)
                        learned_weights_memory = torch.sigmoid(learned_weights_memory)  # [N, 1]
                        learned_weights_memory = learned_weights_memory.permute(1, 0)  # [1, N]
                        
                        # 重新缩放内存权重
                        rescaled_weights_memory = (learned_weights_memory.shape[0] / 
                                                 learned_weights_memory.sum(dim=0)) * learned_weights_memory
                        rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)  # [M, N]
                        distances = distances * rescaled_weights_memory
                
                # 计算该类别的损失
                class_loss = distances.mean()
                total_loss += class_loss
                num_valid_classes += 1
        
        # 返回平均损失
        if num_valid_classes > 0:
            return total_loss / num_valid_classes
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_traditional_losses(self, features: torch.Tensor, 
                                  predictions: torch.Tensor, 
                                  labels: torch.Tensor = None,
                                  is_labeled: bool = True,
                                  margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算传统的类内紧致和类间分离损失
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            margin: 类间分离的最小距离
            
        Returns:
            intra_loss: 类内紧致损失
            inter_loss: 类间分离损失
        """
        if not self.initialized:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        # 计算类原型
        prototypes = {}
        for class_id in range(self.num_classes):
            if self.memory[class_id] is not None:
                memory_features = torch.from_numpy(self.memory[class_id]).to(self.device)
                prototype = torch.mean(memory_features, dim=0)
                prototypes[class_id] = prototype
        
        # 提取当前特征
        current_features = self.extract_high_quality_features(features, predictions, labels, is_labeled)
        
        # 计算类内紧致损失
        intra_loss = torch.tensor(0.0, device=self.device)
        num_valid_intra = 0
        
        for class_id, class_features in current_features.items():
            if class_id in prototypes:
                prototype = prototypes[class_id].detach()
                distances = torch.norm(class_features - prototype.unsqueeze(0), dim=1)
                intra_loss += torch.mean(distances ** 2)
                num_valid_intra += 1
        
        if num_valid_intra > 0:
            intra_loss = intra_loss / num_valid_intra
        
        # 计算类间分离损失
        inter_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        
        if len(prototypes) > 1:
            prototype_list = list(prototypes.values())
            for i in range(len(prototype_list)):
                for j in range(i + 1, len(prototype_list)):
                    distance = torch.norm(prototype_list[i] - prototype_list[j])
                    separation_loss = torch.max(torch.tensor(0.0, device=self.device), 
                                              margin - distance) ** 2
                    inter_loss += separation_loss
                    num_pairs += 1
            
            if num_pairs > 0:
                inter_loss = inter_loss / num_pairs
        
        return intra_loss, inter_loss
    
    def update_and_compute_loss(self, features: torch.Tensor, 
                               predictions: torch.Tensor, 
                               labels: torch.Tensor = None,
                               is_labeled: bool = True,
                               contrastive_weight: float = 1.0,
                               intra_weight: float = 0.1,
                               inter_weight: float = 0.1,
                               margin: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        更新内存并计算损失（SS-Net风格）
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            contrastive_weight: 对比学习损失权重
            intra_weight: 类内紧致损失权重
            inter_weight: 类间分离损失权重
            margin: 类间分离的最小距离
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情
        """
        # 更新内存（不需要梯度）
        with torch.no_grad():
            high_quality_features = self.extract_high_quality_features(
                features, predictions, labels, is_labeled
            )
            
            if len(high_quality_features) > 0:
                self.update_memory(high_quality_features)
        
        # 计算损失（需要梯度）
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        # 1. 对比学习损失（主要损失）
        if self.initialized:
            # 重新提取特征用于损失计算（保持梯度）
            if features.dim() == 5:
                N, C, H, W, D = features.shape
                features_flat = features.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
            else:
                features_flat = features
            
            if predictions.dim() == 5:
                N, num_classes, H, W, D = predictions.shape
                predictions_flat = predictions.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            else:
                predictions_flat = predictions
            
            if labels is not None and labels.dim() == 4:
                labels_flat = labels.view(-1)
            else:
                labels_flat = labels
            
            # 生成类别标签（用于对比学习）
            if is_labeled and labels_flat is not None:
                class_labels = labels_flat
            else:
                # 使用预测作为伪标签
                _, class_labels = torch.max(F.softmax(predictions_flat, dim=1), dim=1)
            
            # 计算对比学习损失
            contrastive_loss = self.contrastive_class_to_class_learned_memory(
                features_flat, class_labels
            )
            
            total_loss += contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()
        else:
            loss_dict['contrastive_loss'] = 0.0
        
        # 2. 传统损失（辅助损失）
        intra_loss, inter_loss = self.compute_traditional_losses(
            features, predictions, labels, is_labeled, margin
        )
        
        total_loss += intra_weight * intra_loss + inter_weight * inter_loss
        loss_dict['intra_loss'] = intra_loss.item()
        loss_dict['inter_loss'] = inter_loss.item()
        loss_dict['total_prototype_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_memory_info(self) -> Dict[str, any]:
        """获取内存信息"""
        info = {
            'initialized': self.initialized,
            'elements_per_class': self.elements_per_class,
            'use_learned_selector': self.use_learned_selector,
            'memory_status': {}
        }
        
        for class_id in range(self.num_classes):
            if self.memory[class_id] is not None:
                info['memory_status'][class_id] = {
                    'num_features': self.memory[class_id].shape[0],
                    'feature_dim': self.memory[class_id].shape[1]
                }
            else:
                info['memory_status'][class_id] = None
        
        return info
    
    def get_selectors(self):
        """获取选择器模块（如果使用学习的选择器）"""
        if self.use_learned_selector:
            return self.feature_selectors, self.memory_selectors
        else:
            return None, None 