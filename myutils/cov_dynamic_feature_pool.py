import torch
import numpy as np
from typing import List, Optional, Tuple
import logging

class CovarianceDynamicFeaturePool:
    """基于二阶统计量的动态特征池 (Cov-DFP)
    
    实现研究方案中提出的基于全局特征协方差分析的多DFP构建与选择框架
    """
    
    def __init__(self, feature_dim: int, num_dfp: int = 8, max_global_features: int = 50000):
        """
        Args:
            feature_dim: 特征维度 D
            num_dfp: 动态特征池数量
            max_global_features: 全局特征池最大容量
        """
        self.feature_dim = feature_dim
        self.num_dfp = num_dfp
        self.max_global_features = max_global_features
        
        # 全局特征池 F_global
        self.global_features = []  # List of tensors
        
        # 多个DFP，每个DFP存储特征列表
        self.dfps = [[] for _ in range(num_dfp)]
        
        # 主要共变模式矩阵 P，形状 [D, num_dfp]
        self.covariation_patterns = None
        
        # 每个DFP的中心特征（均值）
        self.dfp_centers = None
        
        # 是否已完成DFP构建
        self.dfps_built = False
        
    def add_to_global_pool(self, features: torch.Tensor):
        """添加特征到全局特征池
        
        Args:
            features: 形状 [N, D] 的特征张量
        """
        if features.dim() != 2:
            raise ValueError(f"Features should be 2D tensor, got {features.dim()}D")
            
        # 转换为numpy并添加到全局池
        features_np = features.detach().cpu().numpy()
        self.global_features.append(features_np)
        
        # 限制全局池大小
        if self.get_global_pool_size() > self.max_global_features:
            self._trim_global_pool()
    
    def get_global_pool_size(self) -> int:
        """获取全局特征池当前大小"""
        return sum(feat.shape[0] for feat in self.global_features)
    
    def _trim_global_pool(self):
        """裁剪全局特征池到最大容量"""
        total_size = 0
        for i in range(len(self.global_features) - 1, -1, -1):
            total_size += self.global_features[i].shape[0]
            if total_size > self.max_global_features:
                # 裁剪当前特征矩阵
                excess = total_size - self.max_global_features
                self.global_features[i] = self.global_features[i][excess:]
                # 移除更早的特征
                self.global_features = self.global_features[i:]
                break
    
    def build_dfps(self) -> bool:
        """构建多个DFP（阶段二）
        
        Returns:
            bool: 是否成功构建DFP
        """
        if len(self.global_features) == 0:
            logging.warning("Global feature pool is empty, cannot build DFPs")
            return False
            
        # 2.1 计算全局特征的协方差与相关性
        F_global = np.concatenate(self.global_features, axis=0)  # [N, D]
        N, D = F_global.shape
        
        if N < self.num_dfp:
            logging.warning(f"Not enough features ({N}) to build {self.num_dfp} DFPs")
            return False
            
        # 计算均值向量 μ
        mu = np.mean(F_global, axis=0)  # [D]
        
        # 计算协方差矩阵 Σ
        F_centered = F_global - mu[np.newaxis, :]  # [N, D]
        Sigma = np.cov(F_centered.T)  # [D, D]
        
        # 计算相关性矩阵 R
        std = np.sqrt(np.diag(Sigma) + 1e-8)  # [D]
        R = Sigma / (std[:, np.newaxis] * std[np.newaxis, :])  # [D, D]
        
        # 2.2 提取主要的共变模式
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(R)
            # 按特征值降序排列
            idx = np.argsort(eigenvalues)[::-1]
            
            # 选取前num_dfp个特征向量作为主要共变模式
            self.covariation_patterns = eigenvectors[:, idx[:self.num_dfp]]  # [D, num_dfp]
            
        except np.linalg.LinAlgError as e:
            logging.error(f"Eigenvalue decomposition failed: {e}")
            return False
        
        # 2.3 构建动态特征池 (DFP)
        self._assign_features_to_dfps(F_global)
        
        # 2.4 计算每个DFP的中心
        self._compute_dfp_centers()
        
        self.dfps_built = True
        logging.info(f"Successfully built {self.num_dfp} DFPs with {N} global features")
        return True
    
    def _assign_features_to_dfps(self, F_global: np.ndarray):
        """将全局特征分配到不同的DFP中"""
        # 清空现有DFP
        self.dfps = [[] for _ in range(self.num_dfp)]
        
        for i, feature in enumerate(F_global):
            # 计算特征在每个共变模式上的投影绝对值
            projections = np.abs(feature @ self.covariation_patterns)  # [num_dfp]
            
            # 分配到投影绝对值最大的DFP
            best_dfp_idx = np.argmax(projections)
            self.dfps[best_dfp_idx].append(feature)
        
        # 转换为numpy数组
        for j in range(self.num_dfp):
            if len(self.dfps[j]) > 0:
                self.dfps[j] = np.stack(self.dfps[j], axis=0)
            else:
                # 如果某个DFP为空，使用全局特征的子集填充
                logging.warning(f"DFP {j} is empty, filling with global features")
                subset_size = max(1, len(F_global) // self.num_dfp)
                start_idx = j * subset_size
                end_idx = min((j + 1) * subset_size, len(F_global))
                self.dfps[j] = F_global[start_idx:end_idx].copy()
    
    def _compute_dfp_centers(self):
        """计算每个DFP的中心（均值特征）"""
        self.dfp_centers = []
        for dfp_features in self.dfps:
            if len(dfp_features) > 0:
                center = np.mean(dfp_features, axis=0)
                self.dfp_centers.append(center)
            else:
                # 如果DFP为空，使用零向量
                self.dfp_centers.append(np.zeros(self.feature_dim))
        self.dfp_centers = np.stack(self.dfp_centers, axis=0)  # [num_dfp, D]
    
    def get_dfp_target_labels(self, region_features: torch.Tensor) -> torch.Tensor:
        """生成Selector训练的目标标签（阶段二）
        
        Args:
            region_features: 形状 [batch_size, D] 的区域特征
            
        Returns:
            target_labels: 形状 [batch_size] 的目标DFP索引
        """
        if not self.dfps_built:
            raise RuntimeError("DFPs must be built before generating target labels")
            
        features_np = region_features.detach().cpu().numpy()  # [batch_size, D]
        batch_size = features_np.shape[0]
        
        target_labels = []
        for i in range(batch_size):
            feature = features_np[i]  # [D]
            
            # 计算与每个DFP中心的L2距离
            distances = np.linalg.norm(self.dfp_centers - feature[np.newaxis, :], axis=1)  # [num_dfp]
            
            # 选择距离最小的DFP
            target_dfp = np.argmin(distances)
            target_labels.append(target_dfp)
        
        return torch.tensor(target_labels, dtype=torch.long, device=region_features.device)
    
    def get_dfp_features(self, dfp_indices: torch.Tensor, max_features_per_dfp: int = 256) -> List[Optional[torch.Tensor]]:
        """根据Selector预测的DFP索引获取对应的特征
        
        Args:
            dfp_indices: 形状 [batch_size] 的DFP索引
            max_features_per_dfp: 每个DFP最多返回的特征数量
            
        Returns:
            List of feature tensors for each batch item, or None if DFP is empty
        """
        if not self.dfps_built:
            raise RuntimeError("DFPs must be built before accessing DFP features")
            
        batch_size = dfp_indices.shape[0]
        device = dfp_indices.device
        
        result_features = []
        for i in range(batch_size):
            dfp_idx = dfp_indices[i].item()
            
            if 0 <= dfp_idx < self.num_dfp and len(self.dfps[dfp_idx]) > 0:
                dfp_features = self.dfps[dfp_idx]  # [num_features, D]
                
                # 如果特征过多，随机采样
                if len(dfp_features) > max_features_per_dfp:
                    indices = np.random.choice(len(dfp_features), max_features_per_dfp, replace=False)
                    sampled_features = dfp_features[indices]
                else:
                    sampled_features = dfp_features
                
                # 转换为torch tensor
                features_tensor = torch.from_numpy(sampled_features).float().to(device)
                result_features.append(features_tensor)
            else:
                result_features.append(None)
        
        return result_features
    
    def reconstruct_dfps(self) -> bool:
        """重构DFP（可选的周期性操作）
        
        Returns:
            bool: 是否成功重构
        """
        if len(self.global_features) == 0:
            return False
            
        logging.info("Reconstructing DFPs...")
        return self.build_dfps()
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = {
            'global_pool_size': self.get_global_pool_size(),
            'dfps_built': self.dfps_built,
            'num_dfp': self.num_dfp
        }
        
        if self.dfps_built:
            dfp_sizes = [len(dfp) if isinstance(dfp, np.ndarray) else 0 for dfp in self.dfps]
            stats.update({
                'dfp_sizes': dfp_sizes,
                'min_dfp_size': min(dfp_sizes),
                'max_dfp_size': max(dfp_sizes),
                'mean_dfp_size': np.mean(dfp_sizes)
            })
        
        return stats 