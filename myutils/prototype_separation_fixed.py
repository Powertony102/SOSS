#!/usr/bin/env python3
"""
修复版本：Inter-Class Prototype Separation Module for 3D Semi-supervised Medical Image Segmentation

这个模块实现了基于原型的特征分离，以减少半监督学习设置中的类间特征混淆。
它维护类原型并计算类内紧凑性和类间分离损失。

修复内容：
1. 自适应置信度阈值，从低开始逐渐增加
2. 当没有高置信度像素时使用标注像素作为后备
3. 更好地处理单类场景（LA数据集）
4. 改进的日志记录和调试
5. 最小像素要求以确保非零损失

参考：SS-Net (https://github.com/ycwu1997/SS-Net)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import logging


class PrototypeMemoryFixed(nn.Module):
    """
    修复版本的原型内存模块，用于类间特征分离。
    
    这个模块为每个前景类维护类原型 μ_c 并计算：
    - L_intra: 类内紧凑性损失，将同类特征拉近
    - L_inter: 类间分离损失，将不同类原型推远
    
    修复内容：
    - 自适应置信度阈值
    - 标注像素后备机制
    - 更好的单类处理
    - 改进的调试功能
    """
    
    def __init__(
        self,
        num_classes: int,
        feat_dim: Optional[int] = None,
        proto_momentum: float = 0.9,
        conf_thresh: float = 0.5,  # 修复：更低的初始阈值
        conf_thresh_max: float = 0.85,  # 修复：最大阈值
        conf_thresh_rampup: int = 5000,  # 修复：达到最大阈值的迭代数
        update_interval: int = 1,
        lambda_intra: float = 1.0,
        lambda_inter: float = 0.1,
        margin_m: float = 1.0,
        min_pixels_per_class: int = 10,  # 修复：每类最小像素数
        use_labeled_fallback: bool = True,  # 修复：使用标注像素作为后备
        device: str = 'cuda'
    ):
        super(PrototypeMemoryFixed, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.proto_momentum = proto_momentum
        self.conf_thresh_min = conf_thresh
        self.conf_thresh_max = conf_thresh_max
        self.conf_thresh_rampup = conf_thresh_rampup
        self.update_interval = update_interval
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.margin_m = margin_m
        self.min_pixels_per_class = min_pixels_per_class
        self.use_labeled_fallback = use_labeled_fallback
        self.device = device
        
        # 修复：跟踪当前迭代以实现自适应阈值
        self.register_buffer('current_iter', torch.tensor(0, dtype=torch.long))
        
        if self.feat_dim is not None:
            self._initialize_prototype_buffers()
        else:
            self.register_buffer('_buffers_initialized', torch.tensor(False, dtype=torch.bool))
        
        # 统计跟踪
        self.register_buffer('update_count', torch.zeros(num_classes, dtype=torch.long))
        
    def _initialize_prototype_buffers(self):
        """初始化原型相关的buffer，确保在 self.device 上"""
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self.register_buffer('prototypes', torch.zeros(self.num_classes, self.feat_dim, device=device))
        self.register_buffer('prototype_initialized', torch.zeros(self.num_classes, dtype=torch.bool, device=device))
        self.register_buffer('last_update_epoch', torch.tensor(-1, dtype=torch.long, device=device))
        self.register_buffer('_buffers_initialized', torch.tensor(True, dtype=torch.bool, device=device))
    
    def _ensure_buffers_initialized(self, feat_dim: int):
        """确保原型buffers已初始化，如果未初始化则根据输入特征维度初始化，并迁移到输入特征设备"""
        if not hasattr(self, '_buffers_initialized') or not self._buffers_initialized:
            self.feat_dim = feat_dim
            self._initialize_prototype_buffers()
            logging.info(f"PrototypeMemoryFixed: 动态推断特征维度为 {feat_dim}")
        # 保证所有 buffer 在输入特征同设备
        for name in ['prototypes', 'prototype_initialized', 'last_update_epoch', 'update_count', '_buffers_initialized', 'current_iter']:
            buf = getattr(self, name, None)
            if buf is not None:
                device = buf.device
                target_device = torch.device(self.device) if isinstance(self.device, str) else self.device
                if device != target_device:
                    setattr(self, name, buf.to(target_device))
    
    def _get_adaptive_conf_thresh(self) -> float:
        """修复：获取随时间增加的自适应置信度阈值"""
        if self.conf_thresh_rampup <= 0:
            return self.conf_thresh_max
        
        progress = min(1.0, float(self.current_iter) / self.conf_thresh_rampup)
        current_thresh = self.conf_thresh_min + progress * (self.conf_thresh_max - self.conf_thresh_min)
        return current_thresh
    
    def _get_high_confidence_mask_fixed(
        self, 
        pred_flat: torch.Tensor, 
        label_flat: Optional[torch.Tensor] = None,
        is_labelled_flat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Union[int, float, bool]]]:
        """
        修复：生成带有自适应阈值和后备机制的高置信度掩码。
        
        返回:
            conf_mask: (N,) 高置信度预测的布尔掩码
            debug_info: 包含调试信息的字典
        """
        # 获取自适应置信度阈值
        current_thresh = self._get_adaptive_conf_thresh()
        
        # 获取预测置信度和预测类别
        pred_conf, pred_class = torch.max(pred_flat, dim=1)  # (N,)
        
        # 基础置信度掩码（排除背景类0）
        conf_mask = (pred_conf > current_thresh) & (pred_class > 0)
        
        # 对于标注像素，确保预测类别与真实标签匹配
        if label_flat is not None and is_labelled_flat is not None:
            # 仅对标注像素应用标签一致性
            label_consistency = (pred_class == label_flat) | (~is_labelled_flat)
            conf_mask = conf_mask & label_consistency
        
        debug_info = {
            'total_pixels': pred_flat.shape[0],
            'foreground_pixels': (pred_class > 0).sum().item(),
            'confident_pixels': conf_mask.sum().item(),
            'current_thresh': current_thresh,
            'max_confidence': pred_conf.max().item(),
            'mean_confidence': pred_conf.mean().item(),
            'used_labeled_fallback': False
        }
        
        # 修复：当没有找到置信像素时的后备机制
        if not conf_mask.any() and self.use_labeled_fallback:
            if label_flat is not None and is_labelled_flat is not None:
                # 使用标注的前景像素作为后备
                labeled_fg_mask = is_labelled_flat & (label_flat > 0)
                if labeled_fg_mask.any():
                    conf_mask = labeled_fg_mask
                    debug_info['used_labeled_fallback'] = True
                    debug_info['confident_pixels'] = conf_mask.sum().item()
                    logging.info(f"PrototypeMemoryFixed: 使用标注后备，{conf_mask.sum().item()} 个像素")
        
        # 修复：如果可能，确保每类最小像素数
        if conf_mask.any():
            conf_pred_classes = pred_class[conf_mask]
            for class_idx in range(1, self.num_classes + 1):
                class_mask = (conf_pred_classes == class_idx)
                class_count = class_mask.sum().item()
                if class_count < self.min_pixels_per_class:
                    # 尝试为这个类添加更多像素
                    all_class_mask = (pred_class == class_idx)
                    if all_class_mask.sum() >= self.min_pixels_per_class:
                        # 获取这个类的top-k置信像素
                        class_confidences = pred_conf[all_class_mask]
                        _, top_indices = torch.topk(class_confidences, 
                                                  min(self.min_pixels_per_class, len(class_confidences)))
                        class_pixel_indices = torch.where(all_class_mask)[0][top_indices]
                        conf_mask[class_pixel_indices] = True
        
        return conf_mask, debug_info
    
    def _flatten_spatial_dims(
        self, 
        feat: torch.Tensor, 
        pred: torch.Tensor, 
        label: Optional[torch.Tensor] = None,
        is_labelled: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        展平空间维度以进行张量操作。
        
        Args:
            feat: (B, C, H, W, D) 特征张量
            pred: (B, K, H, W, D) 预测张量
            label: (B, 1, H, W, D) 标签张量（可选）
            is_labelled: (B,) 标注掩码（可选）
            
        Returns:
            feat_flat: (B*H*W*D, C)
            pred_flat: (B*H*W*D, K)
            label_flat: (B*H*W*D,) 或 None
            is_labelled_flat: (B*H*W*D,) 或 None
        """
        B, C, H, W, D = feat.shape
        K = pred.shape[1]
        
        # 展平特征: (B, C, H, W, D) -> (B*H*W*D, C)
        feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)
        
        # 展平预测: (B, K, H, W, D) -> (B*H*W*D, K)
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, K)
        
        # 如果提供，展平标签: (B, 1, H, W, D) -> (B*H*W*D,)
        label_flat = None
        if label is not None:
            label_flat = label.view(-1)
        
        # 将is_labelled扩展到空间维度: (B,) -> (B*H*W*D,)
        is_labelled_flat = None
        if is_labelled is not None:
            is_labelled_flat = is_labelled.view(B, 1, 1, 1, 1).expand(B, H, W, D, 1).reshape(-1)
            
        return feat_flat, pred_flat, label_flat, is_labelled_flat
    
    def init_prototypes(
        self, 
        features: torch.Tensor, 
        labels: Optional[torch.Tensor], 
        preds: torch.Tensor, 
        mask: torch.Tensor
    ) -> None:
        """
        使用高置信度特征初始化原型。
        保证所有原型操作在 features.device 上
        """
        if not mask.any():
            return
        actual_feat_dim = features.shape[-1]
        expected_feat_dim = self.prototypes.shape[1]
        if actual_feat_dim != expected_feat_dim:
            raise RuntimeError(
                f"特征维度不匹配！实际输入特征维度: {actual_feat_dim}, "
                f"原型内存期望维度: {expected_feat_dim}. "
                f"请检查PrototypeMemoryFixed初始化时的feat_dim参数是否与模型输出特征维度一致。"
            )
        # 保证 prototypes buffer 在 features.device
        if self.prototypes.device != features.device:
            self.prototypes = self.prototypes.to(features.device)
            self.prototype_initialized = self.prototype_initialized.to(features.device)
            self.last_update_epoch = self.last_update_epoch.to(features.device)
            self.update_count = self.update_count.to(features.device)
        # 获取高置信度像素的预测类别
        _, pred_classes = torch.max(preds[mask], dim=1)  # (M,) where M = mask.sum()
        conf_features = features[mask]  # (M, C)
        for class_idx in range(1, self.num_classes + 1):  # 跳过背景（类别0）
            class_mask = (pred_classes == class_idx)
            if class_mask.any():
                class_features = conf_features[class_mask]  # (N_c, C)
                # 明确迁移到同设备
                mean_proto = torch.mean(class_features, dim=0).detach()
                self.prototypes[class_idx - 1] = mean_proto.to(self.prototypes.device)
                self.prototype_initialized[class_idx - 1] = True
                assert self.prototypes[class_idx - 1].device == class_features.device, '原型与特征设备不一致'
                logging.debug(f"为类别 {class_idx} 初始化原型，使用 "
                            f"{class_mask.sum().item()} 个特征")
    
    def update_prototypes(
        self, 
        features: torch.Tensor, 
        labels: Optional[torch.Tensor], 
        preds: torch.Tensor, 
        mask: torch.Tensor
    ) -> None:
        """
        使用指数移动平均更新原型。
        保证所有原型操作在 features.device 上
        """
        if not mask.any():
            return
        if self.prototypes.device != features.device:
            self.prototypes = self.prototypes.to(features.device)
            self.prototype_initialized = self.prototype_initialized.to(features.device)
            self.last_update_epoch = self.last_update_epoch.to(features.device)
            self.update_count = self.update_count.to(features.device)
        _, pred_classes = torch.max(preds[mask], dim=1)
        conf_features = features[mask]
        for class_idx in range(1, self.num_classes + 1):
            class_mask = (pred_classes == class_idx)
            if class_mask.any():
                class_features = conf_features[class_mask]
                new_prototype = torch.mean(class_features, dim=0).detach()
                new_prototype = new_prototype.to(self.prototypes.device)
                if self.prototype_initialized[class_idx - 1]:
                    old_prototype = self.prototypes[class_idx - 1]
                    self.prototypes[class_idx - 1] = (
                        self.proto_momentum * old_prototype + 
                        (1 - self.proto_momentum) * new_prototype
                    )
                else:
                    self.prototypes[class_idx - 1] = new_prototype
                    self.prototype_initialized[class_idx - 1] = True
                assert self.prototypes[class_idx - 1].device == class_features.device, '原型与特征设备不一致'
                self.update_count[class_idx - 1] += 1
    
    def compute_intra_class_loss(
        self, 
        features: torch.Tensor, 
        preds: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类内紧凑性损失: L_intra = mean(|f_i - μ_{y_i}|^2)
        保证所有原型操作在 features.device 上
        """
        if not mask.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        if self.prototypes.device != features.device:
            self.prototypes = self.prototypes.to(features.device)
        _, pred_classes = torch.max(preds[mask], dim=1)  # (M,)
        conf_features = features[mask]  # (M, C)
        total_loss = 0.0
        valid_pixels = 0
        for class_idx in range(1, self.num_classes + 1):
            if not self.prototype_initialized[class_idx - 1]:
                continue
            class_mask = (pred_classes == class_idx)
            if not class_mask.any():
                continue
            class_features = conf_features[class_mask]  # (N_c, C)
            prototype = self.prototypes[class_idx - 1]
            prototype = prototype.to(class_features.device)
            assert prototype.device == class_features.device, '原型与特征设备不一致'
            distances = torch.norm(class_features - prototype.unsqueeze(0), p=2, dim=1) ** 2
            total_loss += torch.sum(distances)
            valid_pixels += class_features.shape[0]
        if valid_pixels > 0:
            return total_loss / valid_pixels
        else:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
    
    def compute_inter_class_loss(self) -> torch.Tensor:
        """
        计算类间分离损失: L_inter = mean(max(0, margin - |μ_c - μ_c'|)^2)
        
        Returns:
            loss_inter: 标量张量
        """
        if self.lambda_inter == 0.0:
            return torch.tensor(0.0, device=self.device)
            
        # 获取已初始化的原型
        init_mask = self.prototype_initialized
        if init_mask.sum() < 2:
            return torch.tensor(0.0, device=self.device)
            
        prototypes = self.prototypes[init_mask]  # (N_init, C)
        n_init = prototypes.shape[0]
        
        if n_init < 2:
            return torch.tensor(0.0, device=self.device)
        
        # 计算原型之间的成对距离
        # prototypes: (N, C), 扩展为 (N, 1, C) 和 (1, N, C)
        proto_i = prototypes.unsqueeze(1)  # (N, 1, C)
        proto_j = prototypes.unsqueeze(0)  # (1, N, C)
        
        # 计算成对L2距离
        distances = torch.norm(proto_i - proto_j, p=2, dim=2)  # (N, N)
        
        # 创建掩码以排除对角线（i=j）和上三角（避免重复计算）
        mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        
        # 应用基于边际的铰链损失
        margin_violations = torch.clamp(self.margin_m - distances[mask], min=0) ** 2
        
        if margin_violations.numel() > 0:
            return torch.mean(margin_violations)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def forward(
        self, 
        feat: torch.Tensor, 
        label: Optional[torch.Tensor], 
        pred: torch.Tensor, 
        is_labelled: torch.Tensor,
        epoch_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        修复：带有更好调试和自适应行为的前向传播。
        """
        # 修复：更新迭代计数器
        if epoch_idx is not None:
            self.current_iter = torch.tensor(epoch_idx, device=feat.device)
        else:
            self.current_iter += 1
        
        self.device = feat.device
        self._ensure_buffers_initialized(feat.shape[1])
        
        # 确保所有缓冲区在正确设备上
        for name in ['prototypes', 'prototype_initialized', 'last_update_epoch', 'update_count', '_buffers_initialized', 'current_iter']:
            buf = getattr(self, name, None)
            if buf is not None and buf.device != feat.device:
                setattr(self, name, buf.to(feat.device))
        
        # 展平空间维度
        feat_flat, pred_flat, label_flat, is_labelled_flat = self._flatten_spatial_dims(
            feat, pred, label, is_labelled
        )
        
        # 修复：生成带有调试的高置信度掩码
        conf_mask, debug_info = self._get_high_confidence_mask_fixed(pred_flat, label_flat, is_labelled_flat)
        
        # 修复：记录调试信息
        if self.current_iter % 100 == 0:  # 每100次迭代记录一次
            logging.info(f"PrototypeMemoryFixed 调试 (迭代 {self.current_iter}): {debug_info}")
        
        # 如果需要，更新原型
        should_update = True
        if epoch_idx is not None and self.update_interval > 1:
            should_update = (epoch_idx % self.update_interval == 0)
        
        if should_update and conf_mask.any():
            if not self.prototype_initialized.any():
                self.init_prototypes(feat_flat, label_flat, pred_flat, conf_mask)
                logging.info(f"PrototypeMemoryFixed: 使用 {conf_mask.sum().item()} 个像素初始化原型")
            else:
                self.update_prototypes(feat_flat, label_flat, pred_flat, conf_mask)
            if epoch_idx is not None:
                self.last_update_epoch.fill_(epoch_idx)
        
        # 计算损失
        loss_intra = self.compute_intra_class_loss(feat_flat, pred_flat, conf_mask)
        loss_inter = self.compute_inter_class_loss()
        total_loss = self.lambda_intra * loss_intra + self.lambda_inter * loss_inter
        
        # 修复：向返回字典添加调试信息
        result = {
            'intra': loss_intra,
            'inter': loss_inter,
            'total': total_loss,
            'n_confident_pixels': conf_mask.sum().item(),
            'n_initialized_protos': self.prototype_initialized.sum().item(),
            'current_conf_thresh': debug_info['current_thresh'],
            'max_confidence': debug_info['max_confidence'],
            'mean_confidence': debug_info['mean_confidence']
        }
        
        return result
    
    def get_prototype_statistics(self) -> Dict[str, Union[int, float, torch.Tensor]]:
        """获取当前原型的统计信息。"""
        init_mask = self.prototype_initialized
        stats = {
            'num_initialized': int(init_mask.sum().detach().item()),
            'total_classes': int(self.num_classes),
            'update_counts': self.update_count[init_mask].detach().tolist() if init_mask.any() else [],
            'last_update_epoch': int(self.last_update_epoch.detach().item()),
        }
        
        if init_mask.any():
            prototypes = self.prototypes[init_mask]
            proto_norms = torch.norm(prototypes, p=2, dim=1).detach().tolist()
            mean_proto_norm = float(torch.norm(prototypes, p=2, dim=1).mean().detach().item())
            stats.update({
                'prototype_norms': proto_norms,
                'mean_prototype_norm': mean_proto_norm,
            })
            
            # 计算原型之间的成对距离
            if prototypes.shape[0] > 1:
                proto_i = prototypes.unsqueeze(1)
                proto_j = prototypes.unsqueeze(0)
                distances = torch.norm(proto_i - proto_j, p=2, dim=2)
                mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
                pairwise_distances = distances[mask].detach().tolist()
                stats['pairwise_distances'] = pairwise_distances
                stats['mean_pairwise_distance'] = float(distances[mask].mean().detach().item())
        
        return stats