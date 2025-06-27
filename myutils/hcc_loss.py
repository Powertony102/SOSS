import torch
from myutils.covariance_utils import patchwise_covariance, compute_covariance
from myutils.new_correlation_CORAL import coral_loss


def hierarchical_coral(features_s,
                       features_t,
                       weights,
                       cov_mode='patch',
                       patch_size: int = 4,
                       patch_strategy: str = 'mean_cov',  # 'mean_cov'|'mean_loss'|'max_loss'|'topk'
                       topk: int = 5,
                       metric: str = 'fro',
                       scale: float = 1.0):
    """
    计算分层协方差一致性损失 (Hierarchical Covariance Consistency Loss)
    
    Args:
        features_s: List[Tensor] 学生网络特征列表，长度=5，形状 [B,C,D,H,W] 或 [B,C,H,W]
        features_t: List[Tensor] 教师网络特征列表，长度=5，形状 [B,C,D,H,W] 或 [B,C,H,W]
        weights: List[float] 每层的权重系数
        cov_mode: str 协方差计算模式 ('patch' 或 'full')
        patch_size: int patch大小（当cov_mode='patch'时使用）
        patch_strategy: str 补丁策略 ('mean_cov'|'mean_loss'|'max_loss'|'topk')
        topk: int 如果patch_strategy是'topk'，则使用此参数
        metric: str 用于计算CORAL损失的度量
        scale: float 用于缩放协方差矩阵的标量
    
    Returns:
        torch.Tensor: 标量损失值
    """
    if len(features_s) != len(features_t) or len(features_s) != len(weights):
        raise ValueError(f"Length mismatch: features_s={len(features_s)}, features_t={len(features_t)}, weights={len(weights)}")
    
    total_loss = 0.0
    
    for i, (f_s, f_t, w) in enumerate(zip(features_s, features_t, weights)):
        if f_s.size() != f_t.size():
            raise ValueError(f"Feature size mismatch at layer {i}: {f_s.size()} vs {f_t.size()}")
        
        # 处理不同维度的特征图
        if f_s.dim() == 5:  # 3D: [B,C,D,H,W]
            # 将3D特征图重塑为4D以使用现有的patchwise_covariance
            B, C, D, H, W = f_s.shape
            f_s_reshaped = f_s.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
            f_t_reshaped = f_t.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
            
            if cov_mode == 'patch':
                if patch_strategy == 'mean_cov':
                    cov_s = patchwise_covariance(f_s_reshaped, patch_size)
                    cov_t = patchwise_covariance(f_t_reshaped, patch_size)
                    layer_loss = coral_loss(cov_s, cov_t, metric=metric, scale=scale)
                else:
                    # compute per-patch list
                    from myutils.covariance_utils import patchwise_covariance_all
                    covs_s = patchwise_covariance_all(f_s_reshaped, patch_size)  # [P,C,C]
                    covs_t = patchwise_covariance_all(f_t_reshaped, patch_size)
                    per_patch_loss = coral_loss(covs_s, covs_t, metric=metric, scale=scale)  # broadcast
                    if patch_strategy == 'mean_loss':
                        layer_loss = per_patch_loss.mean()
                    elif patch_strategy == 'max_loss':
                        layer_loss = per_patch_loss.max()
                    elif patch_strategy == 'topk':
                        k = min(topk, per_patch_loss.numel())
                        layer_loss = per_patch_loss.topk(k).values.mean()
                    else:
                        raise ValueError(f'Unknown patch_strategy {patch_strategy}')
            else:
                f_s_flat = f_s.flatten(2).permute(0, 2, 1).reshape(-1, C)
                f_t_flat = f_t.flatten(2).permute(0, 2, 1).reshape(-1, C)
                cov_s = compute_covariance(f_s_flat)
                cov_t = compute_covariance(f_t_flat)
                layer_loss = coral_loss(cov_s, cov_t, metric=metric, scale=scale)
        elif f_s.dim() == 4:  # 2D: [B,C,H,W]
            C = f_s.size(1)
            if cov_mode == 'patch':
                if patch_strategy == 'mean_cov':
                    cov_s = patchwise_covariance(f_s, patch_size)
                    cov_t = patchwise_covariance(f_t, patch_size)
                    layer_loss = coral_loss(cov_s, cov_t, metric=metric, scale=scale)
                else:
                    from myutils.covariance_utils import patchwise_covariance_all
                    covs_s = patchwise_covariance_all(f_s, patch_size)
                    covs_t = patchwise_covariance_all(f_t, patch_size)
                    per_patch_loss = coral_loss(covs_s, covs_t, metric=metric, scale=scale)
                    if patch_strategy == 'mean_loss':
                        layer_loss = per_patch_loss.mean()
                    elif patch_strategy == 'max_loss':
                        layer_loss = per_patch_loss.max()
                    elif patch_strategy == 'topk':
                        k = min(topk, per_patch_loss.numel())
                        layer_loss = per_patch_loss.topk(k).values.mean()
                    else:
                        raise ValueError(f'Unknown patch_strategy {patch_strategy}')
            else:
                f_s_flat = f_s.flatten(2).permute(0, 2, 1).reshape(-1, C)
                f_t_flat = f_t.flatten(2).permute(0, 2, 1).reshape(-1, C)
                cov_s = compute_covariance(f_s_flat)
                cov_t = compute_covariance(f_t_flat)
                layer_loss = coral_loss(cov_s, cov_t, metric=metric, scale=scale)
        else:
            raise ValueError(f"Unsupported feature dimension at layer {i}: {f_s.dim()}D")
        
        # 计算CORAL损失并加权累加
        total_loss = total_loss + w * layer_loss
    
    return total_loss


def parse_hcc_weights(weights_str, num_layers=5):
    """
    解析HCC权重字符串
    
    Args:
        weights_str: str 逗号分隔的权重字符串，如 "0.5,0.5,1,1,1.5"
        num_layers: int 期望的层数
    
    Returns:
        List[float]: 权重列表
    """
    try:
        weights = [float(x.strip()) for x in weights_str.split(',')]
        if len(weights) != num_layers:
            raise ValueError(f"Expected {num_layers} weights, got {len(weights)}")
        return weights
    except ValueError as e:
        raise ValueError(f"Invalid HCC weights format '{weights_str}': {e}") 