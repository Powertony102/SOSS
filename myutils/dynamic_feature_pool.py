import torch
import numpy as np
from typing import List, Optional

class DynamicFeaturePool:
    """Second-Order Dynamic Feature Pool (SOGS-style anchors) - GPU版本

    对于每个类别维护一个动态更新的特征集合，并在采样阶段返回该类别相关性矩阵的前 *k* 个特征向量
    作为 second-order anchors。所有操作都在GPU上进行以提高性能。
    """

    def __init__(self, num_labeled_samples: int, num_cls: int, max_store: int = 10000, 
                 ema_alpha: float = 0.9, device: str = 'cuda'):
        """Args
        -----
        num_labeled_samples: 数据集中被标记的图像数（用于索引，仅保留向后兼容）
        num_cls: 类别数
        max_store: 每个类别最多保留的特征数量
        ema_alpha: 更新特征时的 EMA 系数
        device: 计算设备 ('cuda' 或 'cpu')
        """
        self.num_cls = num_cls
        self.max_store = max_store
        self.ema_alpha = ema_alpha
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 为每个类别建立特征列表（GPU张量列表）
        self.class_features = [[] for _ in range(num_cls)]

    # ------------------------------------------------------------------
    # 更新阶段
    # ------------------------------------------------------------------
    def update_labeled_features(self, features_list, labels_list, idxs=None):
        """从当前 batch 更新特征池（GPU版本）。

        Parameters
        ----------
        features_list : List[Tensor]
            len == batch_size，features_list[i] 形状 [N_i, D]
        labels_list : List[Tensor]
            len == batch_size，对应每个像素/voxel 的类别标签 shape [N_i]
        idxs : List[Tensor]
            (保持接口兼容) 图像索引，此实现不使用
        """
        for feat_i, label_i in zip(features_list, labels_list):
            # 确保特征在GPU上
            feat_gpu = feat_i.detach().to(self.device)  # [N_i, D]
            label_gpu = label_i.detach().to(self.device)  # [N_i]

            for c in range(self.num_cls):
                mask_c = label_gpu == c
                if mask_c.any():
                    feats_c = feat_gpu[mask_c]  # [N_c, D]

                    # 追加到类别存储
                    if len(self.class_features[c]) == 0:
                        self.class_features[c].append(feats_c)
                    else:
                        self.class_features[c].append(feats_c)

                    # 拼接并裁剪至 max_store
                    all_feats = torch.cat(self.class_features[c], dim=0)  # [total_N, D]
                    if all_feats.shape[0] > self.max_store:
                        all_feats = all_feats[-self.max_store:]  # 保留最新的特征
                    self.class_features[c] = [all_feats]

    # ------------------------------------------------------------------
    # 采样阶段
    # ------------------------------------------------------------------
    def sample_labeled_features(self, num_eigenvectors: int = 8):
        """返回每个类别的 second-order anchor（相关性矩阵的前 k 个特征向量）- GPU版本。

        Returns
        -------
        List[torch.Tensor] ; len == num_cls
            若某类别无特征或特征不足返回 None，否则返回形状 [k, D] 的GPU张量。
        """
        anchor_list = []
        for c in range(self.num_cls):
            if len(self.class_features[c]) == 0:
                anchor_list.append(None)
                continue

            feats = self.class_features[c][0]  # [N, D] GPU张量
            N, D = feats.shape
            if N < 2:
                anchor_list.append(None)
                continue

            # --- 计算协方差 Σ (GPU) ---
            mu = torch.mean(feats, dim=0, keepdim=True)  # [1, D]
            X_centered = feats - mu  # [N, D]
            cov = torch.mm(X_centered.t(), X_centered) / (N - 1)  # [D, D]

            # --- 转为相关性矩阵 R (GPU) ---
            std = torch.sqrt(torch.diag(cov) + 1e-6)  # [D]
            inv_std = 1.0 / std  # [D]
            R = cov * inv_std.unsqueeze(1) * inv_std.unsqueeze(0)  # [D, D]

            # --- 特征分解 (GPU) ---
            eigvals, eigvecs = torch.linalg.eigh(R)  # ascending order
            idx = torch.argsort(eigvals, descending=True)
            k = min(num_eigenvectors, D)
            top_vecs = eigvecs[:, idx[:k]].t()  # [k, D] GPU张量
            
            # L2归一化anchor向量
            top_vecs_norm = torch.nn.functional.normalize(top_vecs, p=2, dim=1)

            anchor_list.append(top_vecs_norm)

        return anchor_list

    def sample_labeled_features_cpu(self, num_eigenvectors: int = 8):
        """返回每个类别的 second-order anchor（CPU版本，用于向后兼容）。

        Returns
        -------
        List[np.ndarray] ; len == num_cls
            若某类别无特征或特征不足返回 None，否则返回形状 [k, D] 的numpy数组。
        """
        gpu_anchors = self.sample_labeled_features(num_eigenvectors)
        cpu_anchors = []
        
        for anchor in gpu_anchors:
            if anchor is not None:
                cpu_anchors.append(anchor.detach().cpu().numpy())
            else:
                cpu_anchors.append(None)
        
        return cpu_anchors

    def get_class_feature_stats(self) -> dict:
        """获取每个类别的特征统计信息"""
        stats = {}
        for c in range(self.num_cls):
            if len(self.class_features[c]) > 0:
                feats = self.class_features[c][0]
                stats[f'class_{c}'] = {
                    'num_features': feats.shape[0],
                    'feature_dim': feats.shape[1],
                    'device': str(feats.device)
                }
            else:
                stats[f'class_{c}'] = {
                    'num_features': 0,
                    'feature_dim': 0,
                    'device': str(self.device)
                }
        return stats

    def clear_features(self):
        """清空所有特征"""
        self.class_features = [[] for _ in range(self.num_cls)]

    def to_device(self, device: str):
        """将特征移动到指定设备"""
        new_device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if new_device != self.device:
            for c in range(self.num_cls):
                if len(self.class_features[c]) > 0:
                    self.class_features[c][0] = self.class_features[c][0].to(new_device)
            self.device = new_device

# ----------------------------------------------------------------------
# 保持旧接口兼容
# ----------------------------------------------------------------------
SecondOrderDynamicFeaturePool = DynamicFeaturePool
