import torch
import numpy as np

class DynamicFeaturePool:
    """Second-Order Dynamic Feature Pool (SOGS-style anchors).

    对于每个类别维护一个动态更新的特征集合，并在采样阶段返回该类别相关性矩阵的前 *k* 个特征向量
    作为 second-order anchors。
    """

    def __init__(self, num_labeled_samples: int, num_cls: int, max_store: int = 10000, ema_alpha: float = 0.9):
        """Args
        -----
        num_labeled_samples: 数据集中被标记的图像数（用于索引，仅保留向后兼容）
        num_cls: 类别数
        max_store: 每个类别最多保留的特征数量
        ema_alpha: 更新特征时的 EMA 系数
        """
        self.num_cls = num_cls
        self.max_store = max_store
        self.ema_alpha = ema_alpha

        # 为每个类别建立特征列表（numpy数组列表）
        self.class_features = [[] for _ in range(num_cls)]

    # ------------------------------------------------------------------
    # 更新阶段
    # ------------------------------------------------------------------
    def update_labeled_features(self, features_list, labels_list, idxs=None):
        """从当前 batch 更新特征池。

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
            feat_np = feat_i.detach().cpu().numpy()  # [N_i, D]
            label_np = label_i.detach().cpu().numpy()

            for c in range(self.num_cls):
                mask_c = label_np == c
                if np.any(mask_c):
                    feats_c = feat_np[mask_c]  # [N_c, D]

                    # 追加到类别存储
                    if len(self.class_features[c]) == 0:
                        self.class_features[c].append(feats_c)
                    else:
                        self.class_features[c].append(feats_c)

                    # 拼接并裁剪至 max_store
                    all_feats = np.concatenate(self.class_features[c], axis=0)
                    if all_feats.shape[0] > self.max_store:
                        all_feats = all_feats[-self.max_store :]
                    self.class_features[c] = [all_feats]

    # ------------------------------------------------------------------
    # 采样阶段
    # ------------------------------------------------------------------
    def sample_labeled_features(self, num_eigenvectors: int = 8):
        """返回每个类别的 second-order anchor（相关性矩阵的前 k 个特征向量）。

        Returns
        -------
        List[np.ndarray] ; len == num_cls
            若某类别无特征或特征不足返回 None，否则返回形状 [k, D] 的数组。
        """
        anchor_list = []
        for c in range(self.num_cls):
            if len(self.class_features[c]) == 0:
                anchor_list.append(None)
                continue

            feats = self.class_features[c][0]  # [N, D]
            N, D = feats.shape
            if N < 2:
                anchor_list.append(None)
                continue

            # --- 计算协方差 Σ ---
            X = torch.from_numpy(feats).float()
            mu = X.mean(dim=0, keepdim=True)
            X_centered = X - mu
            cov = X_centered.t().mm(X_centered) / (N - 1)

            # --- 转为相关性矩阵 R ---
            std = torch.sqrt(torch.diag(cov) + 1e-6)
            inv_std = 1.0 / std
            R = cov * inv_std.unsqueeze(1) * inv_std.unsqueeze(0)

            # --- 特征分解 ---
            eigvals, eigvecs = torch.linalg.eigh(R)  # ascending order
            idx = torch.argsort(eigvals, descending=True)
            k = min(num_eigenvectors, D)
            top_vecs = eigvecs[:, idx[:k]].t().cpu().numpy()  # [k, D]

            anchor_list.append(top_vecs)

        return anchor_list

# ----------------------------------------------------------------------
# 保持旧接口兼容
# ----------------------------------------------------------------------
SecondOrderDynamicFeaturePool = DynamicFeaturePool
