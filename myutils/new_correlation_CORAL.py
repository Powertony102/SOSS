import torch
import numpy as np

# Local utilities for efficient covariance computation
from myutils.covariance_utils import compute_covariance, patchwise_covariance

def _logm_spd(mat: torch.Tensor, eps: float = 1e-6):
    """Log-Euclidean of an SPD matrix (batched or single)."""
    if mat.dim() == 2:
        eigvals, eigvecs = torch.linalg.eigh(mat)
        eigvals = torch.clamp(eigvals, min=eps)
        return eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.t()
    elif mat.dim() == 3:
        # batch: [N,C,C]
        logs = []
        for m in mat:
            logs.append(_logm_spd(m, eps))
        return torch.stack(logs, dim=0)
    else:
        raise ValueError("mat must be 2-D or 3-D")

def coral_loss(source, target, divide_by_dim: bool = True, scale: float = 1.0, metric: str = 'fro'):
    """CORAL loss with optional log-Euclidean metric.

    Args:
        source, target: covariance matrices ([C,C] or [N,C,C])
        divide_by_dim: divide by d^2 (d=C)
        scale: additional scaling factor
        metric: 'fro' or 'log'
    """
    if metric == 'log':
        source = _logm_spd(source)
        target = _logm_spd(target)

    diff = source - target
    loss = torch.norm(diff, p='fro') ** 2
    if divide_by_dim:
        d = source.size(-1)
        loss = loss / (d * d)
    return scale * loss

def cal_coral_correlation(features,
                          prob,
                          label,
                          memory,
                          num_classes,
                          cov_mode: str = 'full',
                          patch_size: int = 4,
                          temperature: float = 0.1,
                          num_filtered: int = 64):
    correlation_list = []

    for i in range(num_classes):
        class_mask = label == i
        class_feature = features[class_mask, :]
        class_prob = prob[class_mask]
        
        
        # Choose high confidence features
        _, high_conf_indices = torch.sort(class_prob, descending=True)
        high_conf_indices = high_conf_indices[:num_filtered // 4]
        high_conf_features = class_feature[high_conf_indices]
        
        # Choose low confidence features
        low_conf_indices = torch.sort(class_prob, descending=False)[1][:num_filtered // 4]
        low_conf_features = class_feature[low_conf_indices]

        # Merge high confidence and low confidence features
        selected_features = torch.cat((high_conf_features, low_conf_features), dim=0)

        logits_list = []
        for memory_c in memory:
            if memory_c is not None and selected_features.shape[0] > 1 and memory_c.shape[0] > 1:
                memory_c_tensor = torch.from_numpy(memory_c).cuda() if isinstance(memory_c, np.ndarray) else memory_c

                # Calculate covariance matrix based on configuration
                if cov_mode == 'patch' and selected_features.dim() > 2:
                    # If spatial dimensions are retained, use patch-wise covariance
                    source_cov = patchwise_covariance(selected_features, patch_size)
                else:
                    # Fallback to full covariance on flattened features
                    source_cov = compute_covariance(selected_features)

                if cov_mode == 'patch' and memory_c_tensor.dim() > 2:
                    target_cov = patchwise_covariance(memory_c_tensor, patch_size)
                else:
                    target_cov = compute_covariance(memory_c_tensor)

                # Calculate CORAL loss
                covariance_diff = coral_loss(source_cov, target_cov)
                # correlation_score = 1 / (1 + covariance_diff)
                # logits_list.append(correlation_score)
                correlation_list.append(covariance_diff)
        """
        if logits_list:
            logits = torch.stack(logits_list)
            correlation = torch.softmax(logits / temperature, dim=0)
            correlation_list.append(correlation)
        """

    if not correlation_list:
        return [], False
    else:
        correlation_list = torch.stack(correlation_list)  # Ensure the result is a 2D tensor
        return correlation_list, True
