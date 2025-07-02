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
    
    def build_dfps(self, max_optimization_iterations: int = 50, learning_rate: float = 0.01) -> bool:
        """构建多个DFP（阶段二）- 度量学习驱动的构建方案
        
        改进方案：
        1. 协方差分析初始化
        2. 度量学习优化池间分离和池内紧凑
        3. 确保构建阶段就满足度量学习目标
        
        Args:
            max_optimization_iterations: 度量学习优化的最大迭代次数
            learning_rate: 优化学习率
            
        Returns:
            bool: 是否成功构建DFP
        """
        if len(self.global_features) == 0:
            logging.warning("Global feature pool is empty, cannot build DFPs")
            return False
            
        # 获取全局特征
        F_global = torch.cat(self.global_features, dim=0)  # [N, D] GPU张量
        N, D = F_global.shape
        
        if N < self.num_dfp:
            logging.warning(f"Not enough features ({N}) to build {self.num_dfp} DFPs")
            return False
        
        logging.info(f"Building DFPs with metric learning optimization: {N} features, {self.num_dfp} DFPs")
        
        # 第一步：使用协方差分析进行初始化
        success = self._initial_dfp_construction_with_covariance(F_global)
        if not success:
            return False
        
        # 第二步：通过度量学习优化DFP质量
        success = self._optimize_dfps_with_metric_learning(F_global, max_optimization_iterations, learning_rate)
        if not success:
            return False
        
        self.dfps_built = True
        
        # 记录最终统计信息
        stats = self.get_statistics()
        final_compact_loss, final_separate_loss = self._compute_current_metric_losses()
        logging.info(f"DFP construction completed: {stats}")
        logging.info(f"Final metric losses - Compact: {final_compact_loss:.6f}, Separate: {final_separate_loss:.6f}")
        
        return True
    
    def _initial_dfp_construction_with_covariance(self, F_global: torch.Tensor) -> bool:
        """第一步：使用协方差分析进行初始DFP构建"""
        N, D = F_global.shape
        
        # 计算均值向量 μ (GPU)
        mu = torch.mean(F_global, dim=0)  # [D]
        
        # 计算协方差矩阵 Σ (GPU)
        F_centered = F_global - mu.unsqueeze(0)  # [N, D]
        Sigma = torch.mm(F_centered.t(), F_centered) / (N - 1)  # [D, D]
        
        # 计算相关性矩阵 R (GPU)
        std = torch.sqrt(torch.diag(Sigma) + 1e-8)  # [D]
        R = Sigma / (std.unsqueeze(1) * std.unsqueeze(0))  # [D, D]
        
        # 提取主要的共变模式 (GPU)
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
        
        # 构建初始DFP分配
        self._assign_features_to_dfps(F_global)
        
        # 计算初始DFP中心
        self._compute_dfp_centers()
        
        logging.info("Initial DFP construction with covariance analysis completed")
        return True
    
    def _optimize_dfps_with_metric_learning(self, F_global: torch.Tensor, 
                                          max_iterations: int, learning_rate: float) -> bool:
        """第二步：通过度量学习优化DFP分配"""
        N, D = F_global.shape
        
        # 初始化可优化的DFP中心
        initial_centers = []
        for dfp in self.dfps:
            if dfp is not None and dfp.shape[0] > 0:
                center = torch.mean(dfp, dim=0)
                initial_centers.append(center)
            else:
                # 随机初始化空DFP的中心
                center = torch.randn(D, device=self.device) * 0.1
                initial_centers.append(center)
        
        dfp_centers = torch.stack(initial_centers, dim=0)  # [num_dfp, D]
        dfp_centers.requires_grad_(True)
        
        # 优化器
        optimizer = torch.optim.Adam([dfp_centers], lr=learning_rate)
        
        best_loss = float('inf')
        patience = 10
        no_improve_count = 0
        
        logging.info(f"Starting metric learning optimization for {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # 重新分配特征到最近的DFP中心
            distances = torch.cdist(F_global, dfp_centers)  # [N, num_dfp]
            assignments = torch.argmin(distances, dim=1)  # [N]
            
            # 计算池内紧凑性损失
            compact_loss = self._compute_compactness_loss_from_assignments(F_global, assignments, dfp_centers)
            
            # 计算池间分离损失  
            separate_loss = self._compute_separation_loss_from_centers(dfp_centers)
            
            # 总损失
            total_loss = compact_loss + 0.1 * separate_loss  # 分离损失权重较小
            
            total_loss.backward()
            optimizer.step()
            
            # 记录进度
            if iteration % 10 == 0:
                logging.info(f"Optimization iter {iteration}: total_loss={total_loss.item():.6f}, "
                           f"compact={compact_loss.item():.6f}, separate={separate_loss.item():.6f}")
            
            # 早停检查
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logging.info(f"Early stopping at iteration {iteration}")
                    break
        
        # 使用优化后的中心重新分配DFP
        with torch.no_grad():
            distances = torch.cdist(F_global, dfp_centers)
            assignments = torch.argmin(distances, dim=1)
            self._update_dfps_from_assignments(F_global, assignments)
            self._compute_dfp_centers()
        
        logging.info("Metric learning optimization completed")
        return True
    
    def _compute_compactness_loss_from_assignments(self, features: torch.Tensor, 
                                                 assignments: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """计算基于分配的池内紧凑性损失"""
        total_loss = torch.tensor(0.0, device=self.device)
        num_active_pools = 0
        
        for dfp_idx in range(self.num_dfp):
            mask = assignments == dfp_idx
            if mask.any():
                pool_features = features[mask]  # [num_features_in_pool, D]
                pool_center = centers[dfp_idx]  # [D]
                
                # 计算池内特征到中心的距离平方
                distances_squared = torch.sum((pool_features - pool_center.unsqueeze(0)) ** 2, dim=1)
                pool_loss = torch.mean(distances_squared)
                total_loss = total_loss + pool_loss
                num_active_pools += 1
        
        if num_active_pools > 0:
            return total_loss / num_active_pools
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_separation_loss_from_centers(self, centers: torch.Tensor) -> torch.Tensor:
        """计算基于中心的池间分离损失（Softplus版本）"""
        total_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        margin = 1.0  # 适中的margin值
        
        for i in range(self.num_dfp):
            for j in range(i + 1, self.num_dfp):
                center_i = centers[i]  # [D]
                center_j = centers[j]  # [D]
                
                # 计算中心间距离的平方
                distance_squared = torch.sum((center_i - center_j) ** 2)
                
                # Softplus分离损失
                separation_loss = torch.nn.functional.softplus(margin - distance_squared)
                total_loss = total_loss + separation_loss
                num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _update_dfps_from_assignments(self, features: torch.Tensor, assignments: torch.Tensor):
        """根据分配更新DFP内容"""
        dfp_lists = []
        for dfp_idx in range(self.num_dfp):
            mask = assignments == dfp_idx
            if mask.any():
                dfp_features = features[mask].clone()
                dfp_lists.append(dfp_features)
            else:
                # 空DFP，使用全局特征的随机子集
                subset_size = max(1, features.shape[0] // (self.num_dfp * 2))
                random_indices = torch.randperm(features.shape[0], device=self.device)[:subset_size]
                dfp_features = features[random_indices].clone()
                dfp_lists.append(dfp_features)
        
        self.dfps = dfp_lists
    
    def _compute_current_metric_losses(self) -> Tuple[float, float]:
        """计算当前DFP的度量学习损失"""
        if not self.dfps_built:
            return 0.0, 0.0
        
        # 构建虚拟的batch_features_by_dfp用于计算紧凑性损失
        batch_features_by_dfp = {}
        for dfp_idx, dfp in enumerate(self.dfps):
            if dfp is not None and dfp.shape[0] > 0:
                # 取前几个特征作为样本
                sample_size = min(32, dfp.shape[0])
                sample_features = dfp[:sample_size]
                batch_features_by_dfp[dfp_idx] = sample_features
        
        compact_loss = self.compute_intra_pool_compactness_loss(batch_features_by_dfp)
        separate_loss = self.compute_inter_pool_separation_loss(margin=1.0)
        
        return compact_loss.item(), separate_loss.item()
    
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
    
    def reconstruct_dfps(self, max_optimization_iterations: int = 30, learning_rate: float = 0.01) -> bool:
        """重构DFP（可选的周期性操作）- 使用度量学习优化
        
        Args:
            max_optimization_iterations: 重构时的优化迭代次数（相对较少）
            learning_rate: 重构时的学习率
            
        Returns:
            bool: 是否成功重构
        """
        if len(self.global_features) == 0:
            return False
            
        logging.info("Reconstructing DFPs with metric learning optimization...")
        return self.build_dfps(max_optimization_iterations=max_optimization_iterations, 
                              learning_rate=learning_rate)
    
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
    
    def compute_intra_pool_compactness_loss(self, batch_features_by_dfp: dict) -> torch.Tensor:
        """计算池内紧凑性损失 (Intra-Pool Compactness Loss)
        
        公式: L_compact_j = (1/|B_j|) * sum_{f in B_j} ||f - μ_{B_j}||_2^2
        总损失: L_compact = sum_{j s.t. B_j != ∅} L_compact_j
        
        Args:
            batch_features_by_dfp: dict {dfp_idx: features_tensor}
                其中 features_tensor 形状为 [num_features_in_batch, D]
        
        Returns:
            compactness_loss: 标量张量
        """
        if not batch_features_by_dfp:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        active_pools = 0
        
        for dfp_idx, batch_features in batch_features_by_dfp.items():
            if batch_features is None or batch_features.numel() == 0:
                continue
                
            # 确保特征在正确的设备上
            features_gpu = batch_features.to(self.device)  # [num_features_in_batch, D]
            batch_size = features_gpu.shape[0]
            
            if batch_size == 0:
                continue
            
            # 计算批次中心 μ_{B_j}
            batch_center = torch.mean(features_gpu, dim=0)  # [D]
            
            # 计算池内紧凑性损失: (1/|B_j|) * sum_{f in B_j} ||f - μ_{B_j}||_2^2
            distances_squared = torch.sum((features_gpu - batch_center.unsqueeze(0)) ** 2, dim=1)  # [num_features_in_batch]
            pool_compactness_loss = torch.mean(distances_squared)  # 平均距离的平方
            
            total_loss = total_loss + pool_compactness_loss
            active_pools += 1
        
        # 返回平均损失，避免池数量对损失大小的影响
        if active_pools > 0:
            return total_loss / active_pools
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def compute_inter_pool_separation_loss(self, margin: float = 1.0) -> torch.Tensor:
        """计算池间分离性损失 (Inter-Pool Separation Loss) - Softplus 平滑边际版本
        
        改进的公式: L_separate = sum_{i<j} softplus(margin - ||μ_i - μ_j||_2^2)
                             = sum_{i<j} log(1 + exp(margin - ||μ_i - μ_j||_2^2))
        
        优点：
        - 无"死区"：任何距离下都有非零梯度
        - 梯度连续平滑：解决震荡问题
        - 数值稳定：避免硬边界导致的跳跃
        
        Args:
            margin: 分离边际 m，建议使用 0.5-2.0 (相对于距离平方)
        
        Returns:
            separation_loss: 标量张量
        """
        if not self.dfps_built:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        
        # 动态计算每个DFP的当前中心（参与梯度更新）
        current_centers = []
        for i in range(self.num_dfp):
            if self.dfps[i] is not None and self.dfps[i].shape[0] > 0:
                # 计算当前DFP的中心（保持梯度）
                center = torch.mean(self.dfps[i], dim=0)  # [D]
                current_centers.append(center)
            else:
                # 如果DFP为空，跳过此DFP
                current_centers.append(None)
        
        # 计算所有有效池对之间的分离损失
        for i in range(self.num_dfp):
            for j in range(i + 1, self.num_dfp):
                # 只有当两个DFP都非空时才计算分离损失
                if (current_centers[i] is not None and current_centers[j] is not None):
                    
                    center_i = current_centers[i]  # [D]
                    center_j = current_centers[j]  # [D]
                    
                    # 计算两个DFP中心之间的L2距离的平方
                    distance_squared = torch.sum((center_i - center_j) ** 2)
                    
                    # Softplus 平滑分离损失: softplus(margin - distance_squared)
                    # = log(1 + exp(margin - distance_squared))
                    # 这保证了连续的梯度，没有"死区"
                    separation_loss = torch.nn.functional.softplus(margin - distance_squared)
                    
                    total_loss = total_loss + separation_loss
                    num_pairs += 1
        
        # 返回平均损失，加上小的正则化避免数值问题
        if num_pairs > 0:
            avg_loss = total_loss / num_pairs
            # 可选：添加小的数值稳定性处理
            return avg_loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_metric_learning_losses(self, batch_features_by_dfp: dict, 
                                     margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算度量学习相关的两个损失函数
        
        Args:
            batch_features_by_dfp: dict {dfp_idx: features_tensor}
            margin: 池间分离的边际参数
        
        Returns:
            tuple: (intra_pool_compactness_loss, inter_pool_separation_loss)
        """
        compactness_loss = self.compute_intra_pool_compactness_loss(batch_features_by_dfp)
        separation_loss = self.compute_inter_pool_separation_loss(margin)
        
        return compactness_loss, separation_loss
    
    def group_features_by_dfp_predictions(self, features: torch.Tensor, 
                                        dfp_predictions: torch.Tensor) -> dict:
        """根据Selector的预测将特征按DFP分组
        
        Args:
            features: 形状 [batch_size, D] 的特征张量
            dfp_predictions: 形状 [batch_size] 的DFP索引预测
        
        Returns:
            dict: {dfp_idx: features_tensor} 按DFP分组的特征
        """
        batch_features_by_dfp = {}
        
        for dfp_idx in range(self.num_dfp):
            # 找到被分配到当前DFP的特征
            mask = dfp_predictions == dfp_idx
            if mask.any():
                selected_features = features[mask]  # [num_selected, D]
                batch_features_by_dfp[dfp_idx] = selected_features
            else:
                batch_features_by_dfp[dfp_idx] = None
        
        return batch_features_by_dfp
    
    def update_dfps_with_batch_features(self, batch_features_by_dfp: dict, 
                                       update_rate: float = 0.1, max_dfp_size: int = 1000):
        """用批次特征更新DFPs（允许梯度更新）
        
        Args:
            batch_features_by_dfp: dict {dfp_idx: features_tensor}
            update_rate: DFP更新率，控制新特征的权重
            max_dfp_size: DFP的最大大小
        """
        if not self.dfps_built:
            return
            
        for dfp_idx, batch_features in batch_features_by_dfp.items():
            if batch_features is None or batch_features.numel() == 0:
                continue
                
            batch_features_gpu = batch_features.to(self.device).detach()  # 断开梯度以避免累积
            
            if self.dfps[dfp_idx] is None or self.dfps[dfp_idx].shape[0] == 0:
                # 如果DFP为空，直接使用批次特征
                self.dfps[dfp_idx] = batch_features_gpu.clone()
            else:
                # 混合更新：保留一部分老特征，加入一部分新特征
                current_dfp = self.dfps[dfp_idx]
                num_current = current_dfp.shape[0]
                num_new = batch_features_gpu.shape[0]
                
                # 计算保留的老特征数量
                num_keep = min(num_current, max_dfp_size - num_new)
                if num_keep > 0:
                    # 随机选择要保留的老特征
                    keep_indices = torch.randperm(num_current, device=self.device)[:num_keep]
                    kept_features = current_dfp[keep_indices]
                    
                    # 组合老特征和新特征
                    updated_dfp = torch.cat([kept_features, batch_features_gpu], dim=0)
                else:
                    # 如果新特征太多，只保留新特征
                    updated_dfp = batch_features_gpu[:max_dfp_size]
                
                # 确保更新后的DFP不超过最大大小
                if updated_dfp.shape[0] > max_dfp_size:
                    indices = torch.randperm(updated_dfp.shape[0], device=self.device)[:max_dfp_size]
                    updated_dfp = updated_dfp[indices]
                
                self.dfps[dfp_idx] = updated_dfp 