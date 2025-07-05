#!/usr/bin/env python3
"""
FIXED: Inter-Class Prototype Separation Module for 3D Semi-supervised Medical Image Segmentation

This module implements prototype-based feature separation to reduce inter-class feature
confusion in semi-supervised learning settings. It maintains class prototypes and computes
intra-class compactness and inter-class separation losses.

FIXES:
1. Adaptive confidence threshold that starts low and increases over time
2. Fallback mechanism using labeled pixels when no confident pixels exist
3. Better handling of single-class scenarios (LA dataset)
4. Improved logging and debugging
5. Minimum pixel requirements to ensure non-zero losses

Reference: SS-Net (https://github.com/ycwu1997/SS-Net)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import logging


class PrototypeMemoryFixed(nn.Module):
    """
    FIXED Prototype Memory Module for Inter-Class Feature Separation.
    
    This module maintains class prototypes μ_c for each foreground class and computes:
    - L_intra: Intra-class compactness loss to pull same-class features closer
    - L_inter: Inter-class separation loss to push different-class prototypes apart
    
    FIXES:
    - Adaptive confidence threshold
    - Fallback to labeled pixels
    - Better single-class handling
    - Improved debugging
    """
    
    def __init__(
        self,
        num_classes: int,
        feat_dim: Optional[int] = None,
        proto_momentum: float = 0.9,
        conf_thresh: float = 0.5,  # FIXED: Lower initial threshold
        conf_thresh_max: float = 0.85,  # FIXED: Maximum threshold
        conf_thresh_rampup: int = 5000,  # FIXED: Iterations to reach max threshold
        update_interval: int = 1,
        lambda_intra: float = 1.0,
        lambda_inter: float = 0.1,
        margin_m: float = 1.0,
        min_pixels_per_class: int = 10,  # FIXED: Minimum pixels required
        use_labeled_fallback: bool = True,  # FIXED: Use labeled pixels as fallback
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
        
        # FIXED: Track current iteration for adaptive threshold
        self.register_buffer('current_iter', torch.tensor(0, dtype=torch.long))
        
        if self.feat_dim is not None:
            self._initialize_prototype_buffers()
        else:
            self.register_buffer('_buffers_initialized', torch.tensor(False, dtype=torch.bool))
        
        # Statistics tracking
        self.register_buffer('update_count', torch.zeros(num_classes, dtype=torch.long))
        
    def _get_adaptive_conf_thresh(self) -> float:
        """FIXED: Get adaptive confidence threshold that increases over time"""
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
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        FIXED: Generate high-confidence mask with adaptive threshold and fallback.
        
        Returns:
            conf_mask: (N,) boolean mask for high-confidence predictions
            debug_info: Dictionary with debugging information
        """
        # Get adaptive confidence threshold
        current_thresh = self._get_adaptive_conf_thresh()
        
        # Get prediction confidence and predicted class
        pred_conf, pred_class = torch.max(pred_flat, dim=1)  # (N,)
        
        # Base confidence mask (exclude background class 0)
        conf_mask = (pred_conf > current_thresh) & (pred_class > 0)
        
        # For labelled pixels, ensure predicted class matches ground truth
        if label_flat is not None and is_labelled_flat is not None:
            # Only apply label consistency for labelled pixels
            label_consistency = (pred_class == label_flat) | (~is_labelled_flat)
            conf_mask = conf_mask & label_consistency
        
        debug_info = {
            'total_pixels': pred_flat.shape[0],
            'foreground_pixels': (pred_class > 0).sum().item(),
            'confident_pixels': conf_mask.sum().item(),
            'current_thresh': current_thresh,
            'max_confidence': pred_conf.max().item(),
            'mean_confidence': pred_conf.mean().item()
        }
        
        # FIXED: Fallback mechanism when no confident pixels found
        if not conf_mask.any() and self.use_labeled_fallback:
            if label_flat is not None and is_labelled_flat is not None:
                # Use labeled foreground pixels as fallback
                labeled_fg_mask = is_labelled_flat & (label_flat > 0)
                if labeled_fg_mask.any():
                    conf_mask = labeled_fg_mask
                    debug_info['used_labeled_fallback'] = True
                    debug_info['confident_pixels'] = conf_mask.sum().item()
                    logging.info(f"PrototypeMemory: Using labeled fallback, {conf_mask.sum().item()} pixels")
        
        # FIXED: Ensure minimum pixels per class if possible
        if conf_mask.any():
            conf_pred_classes = pred_class[conf_mask]
            for class_idx in range(1, self.num_classes + 1):
                class_mask = (conf_pred_classes == class_idx)
                class_count = class_mask.sum().item()
                if class_count < self.min_pixels_per_class:
                    # Try to add more pixels for this class
                    all_class_mask = (pred_class == class_idx)
                    if all_class_mask.sum() >= self.min_pixels_per_class:
                        # Get top-k confident pixels for this class
                        class_confidences = pred_conf[all_class_mask]
                        _, top_indices = torch.topk(class_confidences, 
                                                  min(self.min_pixels_per_class, len(class_confidences)))
                        class_pixel_indices = torch.where(all_class_mask)[0][top_indices]
                        conf_mask[class_pixel_indices] = True
        
        return conf_mask, debug_info
    
    def forward(
        self, 
        feat: torch.Tensor, 
        label: Optional[torch.Tensor], 
        pred: torch.Tensor, 
        is_labelled: torch.Tensor,
        epoch_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        FIXED: Forward pass with better debugging and adaptive behavior.
        """
        # FIXED: Update iteration counter
        if epoch_idx is not None:
            self.current_iter = torch.tensor(epoch_idx, device=feat.device)
        else:
            self.current_iter += 1
        
        self.device = feat.device
        self._ensure_buffers_initialized(feat.shape[1])
        
        # Ensure all buffers on correct device
        for name in ['prototypes', 'prototype_initialized', 'last_update_epoch', 'update_count', '_buffers_initialized', 'current_iter']:
            buf = getattr(self, name, None)
            if buf is not None and buf.device != feat.device:
                setattr(self, name, buf.to(feat.device))
        
        # Flatten spatial dimensions
        feat_flat, pred_flat, label_flat, is_labelled_flat = self._flatten_spatial_dims(
            feat, pred, label, is_labelled
        )
        
        # FIXED: Generate high-confidence mask with debugging
        conf_mask, debug_info = self._get_high_confidence_mask_fixed(pred_flat, label_flat, is_labelled_flat)
        
        # FIXED: Log debugging information
        if self.current_iter % 100 == 0:  # Log every 100 iterations
            logging.info(f"PrototypeMemory Debug (iter {self.current_iter}): {debug_info}")
        
        # Update prototypes if needed
        should_update = True
        if epoch_idx is not None and self.update_interval > 1:
            should_update = (epoch_idx % self.update_interval == 0)
        
        if should_update and conf_mask.any():
            if not self.prototype_initialized.any():
                self.init_prototypes(feat_flat, label_flat, pred_flat, conf_mask)
                logging.info(f"PrototypeMemory: Initialized prototypes with {conf_mask.sum().item()} pixels")
            else:
                self.update_prototypes(feat_flat, label_flat, pred_flat, conf_mask)
            if epoch_idx is not None:
                self.last_update_epoch.fill_(epoch_idx)
        
        # Compute losses
        loss_intra = self.compute_intra_class_loss(feat_flat, pred_flat, conf_mask)
        loss_inter = self.compute_inter_class_loss()
        total_loss = self.lambda_intra * loss_intra + self.lambda_inter * loss_inter
        
        # FIXED: Add debugging information to return dict
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