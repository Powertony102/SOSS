import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class ImprovedPrototypeManager:
    """
    改进的原型管理器 - 结合 SemiSeg-Contrastive 的设计优势
    参考: https://github.com/Shathe/SemiSeg-Contrastive
    
    主要改进：
    1. 每类保留多个高质量特征而不是单个原型
    2. 在线特征替换策略
    3. 改进的特征质量评估
    4. 更好的梯度管理
    """
    
    def __init__(self, num_classes: int, feature_dim: int, 
                 elements_per_class: int = 32, confidence_threshold: float = 0.8, 
                 device: str = 'cuda'):
        """
        初始化改进的原型管理器
        
        Args:
            num_classes: 类别数量
            feature_dim: 特征维度
            elements_per_class: 每类保留的特征数量
            confidence_threshold: 置信度阈值
            device: 设备
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.elements_per_class = elements_per_class
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # 特征内存 - 每个类别保留多个高质量特征
        self.feature_memory = [None] * num_classes
        self.initialized = False
        
        # 简单的特征质量评估器（可替换为学习的selector）
        self.quality_evaluator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        ).to(device)
        
    def evaluate_feature_quality(self, features: torch.Tensor) -> torch.Tensor:
        """
        评估特征质量
        
        Args:
            features: 特征张量 [N, feature_dim]
            
        Returns:
            quality_scores: 质量分数 [N, 1]
        """
        with torch.no_grad():
            # 使用简单的启发式方法：特征范数 + 学习的质量评估
            norm_scores = torch.norm(features, dim=1, keepdim=True)  # [N, 1]
            norm_scores = norm_scores / (torch.max(norm_scores) + 1e-8)  # 归一化
            
            # 如果需要更复杂的评估，可以启用学习的评估器
            # learned_scores = self.quality_evaluator(features)
            # combined_scores = 0.7 * norm_scores + 0.3 * learned_scores
            
            return norm_scores
    
    def extract_high_quality_features(self, features: torch.Tensor, 
                                    predictions: torch.Tensor, 
                                    labels: torch.Tensor = None,
                                    is_labeled: bool = True) -> Dict[int, torch.Tensor]:
        """
        提取高质量特征 - 改进版本
        
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
        
        # 分离梯度
        features = features.detach()
        predictions = predictions.detach()
        if labels is not None:
            labels = labels.detach()
        
        # 计算预测概率和置信度
        probs = F.softmax(predictions, dim=1)  # [N*H*W*D, num_classes]
        max_probs, pred_classes = torch.max(probs, dim=1)  # [N*H*W*D]
        
        high_quality_features = {}
        
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
                
                # 改进的特征选择：结合置信度和特征质量
                if len(class_features) > 0:
                    # 评估特征质量
                    quality_scores = self.evaluate_feature_quality(class_features).squeeze()  # [N_valid]
                    
                    # 综合评分：置信度 * 特征质量
                    combined_scores = class_confidences * quality_scores
                    
                    # 选择最佳特征
                    if len(class_features) > self.elements_per_class:
                        _, top_indices = torch.topk(combined_scores, self.elements_per_class)
                        selected_features = class_features[top_indices]
                    else:
                        selected_features = class_features
                    
                    high_quality_features[class_id] = selected_features
        
        return high_quality_features
    
    def update_feature_memory(self, features_dict: Dict[int, torch.Tensor]):
        """
        更新特征内存 - 在线替换策略
        
        Args:
            features_dict: 每类新特征的字典
        """
        for class_id, new_features in features_dict.items():
            if class_id == 0:  # 忽略背景类
                continue
                
            if len(new_features) > 0:
                # 在线替换策略：直接用新特征替换旧特征
                self.feature_memory[class_id] = new_features.detach().cpu().numpy()
        
        self.initialized = True
    
    def get_class_prototypes(self) -> Dict[int, torch.Tensor]:
        """
        从特征内存计算类原型
        
        Returns:
            prototypes: 每类的原型向量
        """
        prototypes = {}
        
        for class_id in range(1, self.num_classes):
            if self.feature_memory[class_id] is not None:
                # 计算特征内存中特征的平均值作为原型
                features = torch.from_numpy(self.feature_memory[class_id]).to(self.device)
                prototype = torch.mean(features, dim=0)
                prototypes[class_id] = prototype
        
        return prototypes
    
    def compute_intra_class_loss(self, features: torch.Tensor, 
                                predictions: torch.Tensor, 
                                labels: torch.Tensor = None,
                                is_labeled: bool = True) -> torch.Tensor:
        """
        计算类内紧致损失 - 改进版本
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            
        Returns:
            intra_class_loss: 类内紧致损失
        """
        if not self.initialized:
            return torch.tensor(0.0, device=self.device)
        
        # 提取高质量特征
        high_quality_features = self.extract_high_quality_features(
            features, predictions, labels, is_labeled
        )
        
        # 获取当前原型
        prototypes = self.get_class_prototypes()
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0
        
        for class_id, class_features in high_quality_features.items():
            if class_id in prototypes:
                prototype = prototypes[class_id].detach()  # 不参与梯度计算
                
                # 计算特征到原型的距离
                distances = torch.norm(class_features - prototype.unsqueeze(0), dim=1)
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
        if not self.initialized:
            return torch.tensor(0.0, device=self.device)
        
        prototypes = self.get_class_prototypes()
        
        if len(prototypes) < 2:
            return torch.tensor(0.0, device=self.device)
        
        prototype_list = list(prototypes.values())
        total_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        
        # 计算所有类别对之间的分离损失
        for i in range(len(prototype_list)):
            for j in range(i + 1, len(prototype_list)):
                prototype_i = prototype_list[i].detach()
                prototype_j = prototype_list[j].detach()
                
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
    
    def compute_contrastive_loss(self, features: torch.Tensor, 
                               predictions: torch.Tensor, 
                               labels: torch.Tensor = None,
                               is_labeled: bool = True,
                               temperature: float = 0.1) -> torch.Tensor:
        """
        计算对比学习损失 - 基于特征内存
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            temperature: 温度参数
            
        Returns:
            contrastive_loss: 对比学习损失
        """
        if not self.initialized:
            return torch.tensor(0.0, device=self.device)
        
        # 提取高质量特征
        high_quality_features = self.extract_high_quality_features(
            features, predictions, labels, is_labeled
        )
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0
        
        for class_id, class_features in high_quality_features.items():
            if self.feature_memory[class_id] is not None:
                # 获取内存中的正样本
                memory_features = torch.from_numpy(self.feature_memory[class_id]).to(self.device)
                
                # 计算当前特征与内存特征的相似性
                similarities = torch.mm(F.normalize(class_features, dim=1), 
                                      F.normalize(memory_features, dim=1).t())
                
                # 计算对比损失（简化版本）
                pos_similarities = torch.max(similarities, dim=1)[0]
                contrastive_loss = -torch.log(torch.exp(pos_similarities / temperature).mean())
                
                total_loss += contrastive_loss
                num_valid += 1
        
        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=self.device)
    
    def update_and_compute_loss(self, features: torch.Tensor, 
                               predictions: torch.Tensor, 
                               labels: torch.Tensor = None,
                               is_labeled: bool = True,
                               intra_weight: float = 1.0,
                               inter_weight: float = 0.1,
                               contrastive_weight: float = 0.5,
                               margin: float = 1.0,
                               temperature: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        更新特征内存并计算损失（一体化接口）
        
        Args:
            features: 特征张量
            predictions: 预测概率
            labels: 真实标签
            is_labeled: 是否为标记数据
            intra_weight: 类内紧致损失权重
            inter_weight: 类间分离损失权重
            contrastive_weight: 对比学习损失权重
            margin: 类间分离的最小距离
            temperature: 对比学习温度参数
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情
        """
        # 提取高质量特征用于更新内存（不需要梯度）
        with torch.no_grad():
            high_quality_features = self.extract_high_quality_features(
                features, predictions, labels, is_labeled
            )
            
            # 更新特征内存
            if len(high_quality_features) > 0:
                self.update_feature_memory(high_quality_features)
        
        # 计算损失（需要梯度）
        intra_loss = self.compute_intra_class_loss(features, predictions, labels, is_labeled)
        inter_loss = self.compute_inter_class_loss(margin)
        contrastive_loss = self.compute_contrastive_loss(features, predictions, labels, is_labeled, temperature)
        
        # 总损失
        total_loss = (intra_weight * intra_loss + 
                     inter_weight * inter_loss + 
                     contrastive_weight * contrastive_loss)
        
        loss_dict = {
            'intra_loss': intra_loss.item(),
            'inter_loss': inter_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_prototype_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def get_memory_info(self) -> Dict[str, any]:
        """获取特征内存信息"""
        info = {
            'initialized': self.initialized,
            'elements_per_class': self.elements_per_class,
            'memory_status': {}
        }
        
        for class_id in range(1, self.num_classes):
            if self.feature_memory[class_id] is not None:
                info['memory_status'][class_id] = {
                    'num_features': self.feature_memory[class_id].shape[0],
                    'feature_dim': self.feature_memory[class_id].shape[1]
                }
            else:
                info['memory_status'][class_id] = None
        
        # 计算类间距离统计
        prototypes = self.get_class_prototypes()
        if len(prototypes) > 1:
            prototype_list = list(prototypes.values())
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