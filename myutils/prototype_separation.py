#!/usr/bin/env python3
"""
Inter-Class Prototype Separation Module for 3D Semi-supervised Medical Image Segmentation

This module implements prototype-based feature separation to reduce inter-class feature
confusion in semi-supervised learning settings. It maintains class prototypes and computes
intra-class compactness and inter-class separation losses.

Reference: SS-Net (https://github.com/ycwu1997/SS-Net)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import logging


class PrototypeMemory(nn.Module):
    """
    Prototype Memory Module for Inter-Class Feature Separation.
    
    This module maintains class prototypes μ_c for each foreground class and computes:
    - L_intra: Intra-class compactness loss to pull same-class features closer
    - L_inter: Inter-class separation loss to push different-class prototypes apart
    
    Args:
        num_classes (int): Number of foreground classes (excluding background)
        feat_dim (int): Feature dimension
        proto_momentum (float): Momentum for prototype updates (0.0=recompute, 1.0=no update)
        conf_thresh (float): Confidence threshold for high-quality predictions
        update_interval (int): Update prototypes every N epochs
        lambda_intra (float): Weight for intra-class compactness loss
        lambda_inter (float): Weight for inter-class separation loss (0 to disable)
        margin_m (float): Minimum distance margin for inter-class separation
        device (str): Device to place tensors on
    """
    
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        proto_momentum: float = 0.9,
        conf_thresh: float = 0.8,
        update_interval: int = 1,
        lambda_intra: float = 1.0,
        lambda_inter: float = 0.1,
        margin_m: float = 1.0,
        device: str = 'cuda'
    ):
        super(PrototypeMemory, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.proto_momentum = proto_momentum
        self.conf_thresh = conf_thresh
        self.update_interval = update_interval
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.margin_m = margin_m
        self.device = device
        
        # Register prototype buffers - shape: (num_classes, feat_dim)
        # Note: class 0 is background, so we store prototypes for classes 1 to num_classes
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('prototype_initialized', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('last_update_epoch', torch.tensor(-1, dtype=torch.long))
        
        # Statistics tracking
        self.register_buffer('update_count', torch.zeros(num_classes, dtype=torch.long))
        
    def _get_high_confidence_mask(
        self, 
        pred_flat: torch.Tensor, 
        label_flat: Optional[torch.Tensor] = None,
        is_labelled_flat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate high-confidence mask for reliable feature extraction.
        
        Args:
            pred_flat: (N, K) flattened predictions
            label_flat: (N,) flattened labels (optional)
            is_labelled_flat: (N,) mask indicating labelled pixels (optional)
            
        Returns:
            conf_mask: (N,) boolean mask for high-confidence predictions
        """
        # Get prediction confidence and predicted class
        pred_conf, pred_class = torch.max(pred_flat, dim=1)  # (N,)
        
        # Base confidence mask (exclude background class 0)
        conf_mask = (pred_conf > self.conf_thresh) & (pred_class > 0)
        
        # For labelled pixels, ensure predicted class matches ground truth
        if label_flat is not None and is_labelled_flat is not None:
            # Only apply label consistency for labelled pixels
            label_consistency = (pred_class == label_flat) | (~is_labelled_flat)
            conf_mask = conf_mask & label_consistency
            
        return conf_mask
    
    def _flatten_spatial_dims(
        self, 
        feat: torch.Tensor, 
        pred: torch.Tensor, 
        label: Optional[torch.Tensor] = None,
        is_labelled: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Flatten spatial dimensions for tensor operations.
        
        Args:
            feat: (B, C, H, W, D) feature tensor
            pred: (B, K, H, W, D) prediction tensor
            label: (B, 1, H, W, D) label tensor (optional)
            is_labelled: (B,) labelled mask (optional)
            
        Returns:
            feat_flat: (B*H*W*D, C)
            pred_flat: (B*H*W*D, K)
            label_flat: (B*H*W*D,) or None
            is_labelled_flat: (B*H*W*D,) or None
        """
        B, C, H, W, D = feat.shape
        K = pred.shape[1]
        
        # Flatten features: (B, C, H, W, D) -> (B*H*W*D, C)
        feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)
        
        # Flatten predictions: (B, K, H, W, D) -> (B*H*W*D, K)
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, K)
        
        # Flatten labels if provided: (B, 1, H, W, D) -> (B*H*W*D,)
        label_flat = None
        if label is not None:
            label_flat = label.view(-1)
        
        # Expand is_labelled to spatial dimensions: (B,) -> (B*H*W*D,)
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
        Initialize prototypes using high-confidence features.
        
        Args:
            features: (N, C) flattened features
            labels: (N,) flattened labels (optional)
            preds: (N, K) flattened predictions
            mask: (N,) high-confidence mask
        """
        if not mask.any():
            return
            
        # Get predicted classes for high-confidence pixels
        _, pred_classes = torch.max(preds[mask], dim=1)  # (M,) where M = mask.sum()
        conf_features = features[mask]  # (M, C)
        
        # Initialize prototypes for each class
        for class_idx in range(1, self.num_classes + 1):  # Skip background (class 0)
            class_mask = (pred_classes == class_idx)
            
            if class_mask.any():
                # Use mean of high-confidence features for this class
                class_features = conf_features[class_mask]  # (N_c, C)
                self.prototypes[class_idx - 1] = torch.mean(class_features, dim=0)
                self.prototype_initialized[class_idx - 1] = True
                
                logging.debug(f"Initialized prototype for class {class_idx} with "
                            f"{class_mask.sum().item()} features")
    
    def update_prototypes(
        self, 
        features: torch.Tensor, 
        labels: Optional[torch.Tensor], 
        preds: torch.Tensor, 
        mask: torch.Tensor
    ) -> None:
        """
        Update prototypes using exponential moving average.
        
        Args:
            features: (N, C) flattened features
            labels: (N,) flattened labels (optional)
            preds: (N, K) flattened predictions
            mask: (N,) high-confidence mask
        """
        if not mask.any():
            return
            
        # Get predicted classes for high-confidence pixels
        _, pred_classes = torch.max(preds[mask], dim=1)
        conf_features = features[mask]
        
        # Update prototypes for each class
        for class_idx in range(1, self.num_classes + 1):
            class_mask = (pred_classes == class_idx)
            
            if class_mask.any():
                # Compute new prototype from current batch
                class_features = conf_features[class_mask]
                new_prototype = torch.mean(class_features, dim=0)
                
                if self.prototype_initialized[class_idx - 1]:
                    # Exponential moving average update
                    old_prototype = self.prototypes[class_idx - 1]
                    self.prototypes[class_idx - 1] = (
                        self.proto_momentum * old_prototype + 
                        (1 - self.proto_momentum) * new_prototype
                    )
                else:
                    # First initialization
                    self.prototypes[class_idx - 1] = new_prototype
                    self.prototype_initialized[class_idx - 1] = True
                    
                self.update_count[class_idx - 1] += 1
    
    def compute_intra_class_loss(
        self, 
        features: torch.Tensor, 
        preds: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intra-class compactness loss: L_intra = mean(|f_i - μ_{y_i}|^2)
        
        Args:
            features: (N, C) flattened features
            preds: (N, K) flattened predictions
            mask: (N,) high-confidence mask
            
        Returns:
            loss_intra: scalar tensor
        """
        if not mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # Get predicted classes and features for high-confidence pixels
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
                
            # Get features for this class
            class_features = conf_features[class_mask]  # (N_c, C)
            prototype = self.prototypes[class_idx - 1]  # (C,)
            
            # Compute squared L2 distance to prototype
            distances = torch.norm(class_features - prototype.unsqueeze(0), p=2, dim=1) ** 2
            total_loss += torch.sum(distances)
            valid_pixels += class_features.shape[0]
            
        if valid_pixels > 0:
            return total_loss / valid_pixels
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def compute_inter_class_loss(self) -> torch.Tensor:
        """
        Compute inter-class separation loss: L_inter = mean(max(0, margin - |μ_c - μ_c'|)^2)
        
        Returns:
            loss_inter: scalar tensor
        """
        if self.lambda_inter == 0.0:
            return torch.tensor(0.0, device=self.device)
            
        # Get initialized prototypes
        init_mask = self.prototype_initialized
        if init_mask.sum() < 2:
            return torch.tensor(0.0, device=self.device)
            
        prototypes = self.prototypes[init_mask]  # (N_init, C)
        n_init = prototypes.shape[0]
        
        if n_init < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute pairwise distances between prototypes
        # prototypes: (N, C), expand to (N, 1, C) and (1, N, C)
        proto_i = prototypes.unsqueeze(1)  # (N, 1, C)
        proto_j = prototypes.unsqueeze(0)  # (1, N, C)
        
        # Compute pairwise L2 distances
        distances = torch.norm(proto_i - proto_j, p=2, dim=2)  # (N, N)
        
        # Create mask to exclude diagonal (i=j) and upper triangle (avoid double counting)
        mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        
        # Apply margin-based hinge loss
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
        Forward pass to compute prototype-based losses.
        
        Args:
            feat: (B, C, H, W, D) decoder features
            label: (B, 1, H, W, D) ground truth labels (None for unlabelled)
            pred: (B, K, H, W, D) softmax predictions
            is_labelled: (B,) boolean mask indicating labelled samples
            epoch_idx: current epoch index for update scheduling
            
        Returns:
            loss_dict: Dictionary containing 'intra', 'inter', and 'total' losses
        """
        # Flatten spatial dimensions
        feat_flat, pred_flat, label_flat, is_labelled_flat = self._flatten_spatial_dims(
            feat, pred, label, is_labelled
        )
        
        # Generate high-confidence mask
        conf_mask = self._get_high_confidence_mask(pred_flat, label_flat, is_labelled_flat)
        
        # Update prototypes if needed
        should_update = True
        if epoch_idx is not None and self.update_interval > 1:
            should_update = (epoch_idx % self.update_interval == 0)
            
        if should_update and conf_mask.any():
            if not self.prototype_initialized.any():
                # Initial prototype computation
                self.init_prototypes(feat_flat, label_flat, pred_flat, conf_mask)
            else:
                # Update existing prototypes
                self.update_prototypes(feat_flat, label_flat, pred_flat, conf_mask)
                
            if epoch_idx is not None:
                self.last_update_epoch.fill_(epoch_idx)
        
        # Compute losses
        loss_intra = self.compute_intra_class_loss(feat_flat, pred_flat, conf_mask)
        loss_inter = self.compute_inter_class_loss()
        
        # Combine losses
        total_loss = self.lambda_intra * loss_intra + self.lambda_inter * loss_inter
        
        return {
            'intra': loss_intra,
            'inter': loss_inter,
            'total': total_loss,
            'n_confident_pixels': conf_mask.sum().item(),
            'n_initialized_protos': self.prototype_initialized.sum().item()
        }
    
    def get_prototype_statistics(self) -> Dict[str, Union[int, float, torch.Tensor]]:
        """Get statistics about current prototypes."""
        init_mask = self.prototype_initialized
        stats = {
            'num_initialized': init_mask.sum().item(),
            'total_classes': self.num_classes,
            'update_counts': self.update_count[init_mask].detach().cpu().numpy() if init_mask.any() else [],
            'last_update_epoch': self.last_update_epoch.item(),
        }
        
        if init_mask.any():
            prototypes = self.prototypes[init_mask]
            stats.update({
                'prototype_norms': torch.norm(prototypes, p=2, dim=1).detach().cpu().numpy(),
                'mean_prototype_norm': torch.norm(prototypes, p=2, dim=1).mean().item(),
            })
            
            # Compute pairwise distances between prototypes
            if prototypes.shape[0] > 1:
                proto_i = prototypes.unsqueeze(1)
                proto_j = prototypes.unsqueeze(0)
                distances = torch.norm(proto_i - proto_j, p=2, dim=2)
                mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
                stats['pairwise_distances'] = distances[mask].detach().cpu().numpy()
                stats['mean_pairwise_distance'] = distances[mask].mean().item()
        
        return stats 