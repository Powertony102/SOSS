import torch
import numpy as np
from typing import List, Optional, Tuple
import logging

class CovarianceDynamicFeaturePool:
    """基于二阶统计量的动态特征池 (Cov-DFP)
    
    实现研究方案中提出的基于全局特征协方差分析的多DFP构建与选择框架
    支持GPU加速计算
    """
    
    def __init__(self, feature_dim: int, num_dfp: int = 8, max_global_features: int = 50000, device: str = 'cuda'):
        """
        Args:
            feature_dim: 特征维度 D
            num_dfp: 动态特征池数量
            max_global_features: 全局特征池最大容量
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.feature_dim = feature_dim
        self.num_dfp = num_dfp
        self.max_global_features = max_global_features
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 全局特征池 F_global (存储为GPU张量列表)
        self.global_features = []  # List of GPU tensors
        
        # 多个DFP，每个DFP存储特征张量
        self.dfps = [None for _ in range(num_dfp)]
        
        # 主要共变模式矩阵 P，形状 [D, num_dfp] (GPU张量)
        self.covariation_patterns = None
        
        # 每个DFP的中心特征（均值）(GPU张量)
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
            
        # 确保特征在正确的设备上并detach
        features_gpu = features.detach().to(self.device)
        self.global_features.append(features_gpu)
        
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
        """构建多个DFP（阶段二）- GPU加速版本
        
        Returns:
            bool: 是否成功构建DFP
        """
        if len(self.global_features) == 0:
            logging.warning("Global feature pool is empty, cannot build DFPs")
            return False
            
        # 2.1 计算全局特征的协方差与相关性 (GPU加速)
        F_global = torch.cat(self.global_features, dim=0)  # [N, D] GPU张量
        N, D = F_global.shape
        
        if N < self.num_dfp:
            logging.warning(f"Not enough features ({N}) to build {self.num_dfp} DFPs")
            return False
            
        # 计算均值向量 μ (GPU)
        mu = torch.mean(F_global, dim=0)  # [D]
        
        # 计算协方差矩阵 Σ (GPU)
        F_centered = F_global - mu.unsqueeze(0)  # [N, D]
        Sigma = torch.mm(F_centered.t(), F_centered) / (N - 1)  # [D, D]
        
        # 计算相关性矩阵 R (GPU)
        std = torch.sqrt(torch.diag(Sigma) + 1e-8)  # [D]
        R = Sigma / (std.unsqueeze(1) * std.unsqueeze(0))  # [D, D]
        
        # 2.2 提取主要的共变模式 (GPU)
        try:
            # 使用torch的特征分解 (GPU加速)
            eigenvalues, eigenvectors = torch.linalg.eigh(R)
            # 按特征值降序排列
            idx = torch.argsort(eigenvalues, descending=True)
            
            # 选取前num_dfp个特征向量作为主要共变模式
            self.covariation_patterns = eigenvectors[:, idx[:self.num_dfp]]  # [D, num_dfp] GPU张量
            
        except Exception as e:
            logging.error(f"Eigenvalue decomposition failed: {e}")
            return False
        
        # 2.3 构建动态特征池 (DFP) (GPU)
        self._assign_features_to_dfps(F_global)
        
        # 2.4 计算每个DFP的中心 (GPU)
        self._compute_dfp_centers()
        
        self.dfps_built = True
        logging.info(f"Successfully built {self.num_dfp} DFPs with {N} global features on {self.device}")
        return True
    
    def _assign_features_to_dfps(self, F_global: torch.Tensor):
        """将全局特征分配到不同的DFP中 (GPU版本)"""
        # 初始化DFP列表
        dfp_lists = [[] for _ in range(self.num_dfp)]
        
        # 计算所有特征在共变模式上的投影 (批量GPU计算)
        projections = torch.abs(torch.mm(F_global, self.covariation_patterns))  # [N, num_dfp]
        
        # 找到每个特征最佳的DFP
        best_dfp_indices = torch.argmax(projections, dim=1)  # [N]
        
        # 将特征分配到对应的DFP
        for dfp_idx in range(self.num_dfp):
            mask = best_dfp_indices == dfp_idx
            if mask.any():
                dfp_lists[dfp_idx] = F_global[mask]  # 直接获取GPU张量子集
            else:
                # 如果某个DFP为空，使用全局特征的子集填充
                logging.warning(f"DFP {dfp_idx} is empty, filling with global features")
                subset_size = max(1, F_global.shape[0] // self.num_dfp)
                start_idx = dfp_idx * subset_size
                end_idx = min((dfp_idx + 1) * subset_size, F_global.shape[0])
                dfp_lists[dfp_idx] = F_global[start_idx:end_idx].clone()
        
        # 存储为GPU张量
        self.dfps = dfp_lists
    
    def _compute_dfp_centers(self):
        """计算每个DFP的中心（均值特征）(GPU版本)"""
        dfp_centers_list = []
        for dfp_features in self.dfps:
            if dfp_features is not None and dfp_features.shape[0] > 0:
                center = torch.mean(dfp_features, dim=0)  # GPU上计算均值
                dfp_centers_list.append(center)
            else:
                # 如果DFP为空，使用零向量
                center = torch.zeros(self.feature_dim, device=self.device)
                dfp_centers_list.append(center)
        self.dfp_centers = torch.stack(dfp_centers_list, dim=0)  # [num_dfp, D] GPU张量
    
    def get_dfp_target_labels(self, region_features: torch.Tensor) -> torch.Tensor:
        """生成Selector训练的目标标签（阶段二）- GPU版本
        
        Args:
            region_features: 形状 [batch_size, D] 的区域特征
            
        Returns:
            target_labels: 形状 [batch_size] 的目标DFP索引
        """
        if not self.dfps_built:
            raise RuntimeError("DFPs must be built before generating target labels")
            
        # 确保特征在同一设备上
        features_gpu = region_features.to(self.device)  # [batch_size, D]
        
        # 计算与每个DFP中心的L2距离 (批量GPU计算)
        # features_gpu: [batch_size, D], dfp_centers: [num_dfp, D]
        # 使用广播计算距离矩阵
        distances = torch.norm(features_gpu.unsqueeze(1) - self.dfp_centers.unsqueeze(0), dim=2)  # [batch_size, num_dfp]
        
        # 选择距离最小的DFP
        target_labels = torch.argmin(distances, dim=1)  # [batch_size]
        
        return target_labels.to(region_features.device)
    
    def get_dfp_features(self, dfp_indices: torch.Tensor, max_features_per_dfp: int = 256) -> List[Optional[torch.Tensor]]:
        """根据Selector预测的DFP索引获取对应的特征 - GPU版本
        
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
            
            if 0 <= dfp_idx < self.num_dfp and self.dfps[dfp_idx] is not None and self.dfps[dfp_idx].shape[0] > 0:
                dfp_features = self.dfps[dfp_idx]  # [num_features, D] GPU张量
                
                # 如果特征过多，随机采样 (GPU上进行)
                if dfp_features.shape[0] > max_features_per_dfp:
                    # 使用torch在GPU上进行随机采样
                    indices = torch.randperm(dfp_features.shape[0], device=self.device)[:max_features_per_dfp]
                    sampled_features = dfp_features[indices]
                else:
                    sampled_features = dfp_features
                
                # 转移到目标设备
                features_tensor = sampled_features.to(device)
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
            'num_dfp': self.num_dfp,
            'device': str(self.device)
        }
        
        if self.dfps_built:
            dfp_sizes = [dfp.shape[0] if dfp is not None else 0 for dfp in self.dfps]
            stats.update({
                'dfp_sizes': dfp_sizes,
                'min_dfp_size': min(dfp_sizes),
                'max_dfp_size': max(dfp_sizes),
                'mean_dfp_size': float(torch.tensor(dfp_sizes, dtype=torch.float).mean().item())
            })
        
        return stats 