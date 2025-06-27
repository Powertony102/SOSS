import torch
import torch.nn.functional as F


def compute_covariance(X: torch.Tensor) -> torch.Tensor:
    """Compute unbiased covariance matrix of samples.

    Args:
        X: Tensor of shape [N, C] where N is number of samples and C feature dimension.

    Returns:
        Covariance matrix of shape [C, C] (same device/dtype as X).
    """
    if X.dim() != 2:
        raise ValueError("compute_covariance expects a 2-D tensor [N, C]")
    N = X.size(0)
    if N < 2:
        # Degenerate case – return zeros
        return torch.zeros(X.size(1), X.size(1), device=X.device, dtype=X.dtype)

    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = X_centered.t().matmul(X_centered) / (N - 1)
    return cov


def _patch_covariance_batch(patches: torch.Tensor) -> torch.Tensor:
    """Helper to compute covariance for a batch of patches.

    Args:
        patches: Tensor shape [B, N, C] where B is number of patches,
                 N = k^2 (pixels per patch), C channels.
    Returns:
        Tensor [B, C, C] – covariance per patch.
    """
    # Center
    patches_centered = patches - patches.mean(dim=1, keepdim=True)
    # Compute covariance with einsum for efficiency
    cov = torch.einsum('bnc,bnd->bcd', patches_centered, patches_centered) / (patches.size(1) - 1 + 1e-5)
    return cov


def patchwise_covariance(feat: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """Patch-wise covariance averaged over all patches.

    Currently supports 4-D input [B, C, H, W]. If tensor has other dims,
    it falls back to full covariance on flattened spatial dims.

    Args:
        feat: Tensor of shape [B, C, H, W].
        patch_size: Spatial size of square patch (k).
    Returns:
        Covariance matrix [C, C] averaged over all patches in the batch.
    """
    if feat.dim() != 4:
        # Fallback – flatten spatial dims
        B, *rest = feat.shape
        feat_flat = feat.view(B, feat.size(1), -1).permute(0, 2, 1).reshape(-1, feat.size(1))
        return compute_covariance(feat_flat)

    B, C, H, W = feat.shape
    # Make sure H and W are divisible by patch_size; else pad to fit
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h or pad_w:
        feat = F.pad(feat, (0, pad_w, 0, pad_h), mode='replicate')
        H, W = H + pad_h, W + pad_w

    unfold = F.unfold(feat, kernel_size=patch_size, stride=patch_size)  # [B, C*k*k, L]
    k2 = patch_size * patch_size
    L = unfold.size(-1)
    patches = unfold.view(B, C, k2, L).permute(0, 3, 2, 1).contiguous()  # [B, L, k2, C]
    patches = patches.view(-1, k2, C)  # [B*L, k2, C]

    cov_per_patch = _patch_covariance_batch(patches)  # [B*L, C, C]
    cov_mean = cov_per_patch.mean(dim=0)  # average across patches and batch
    return cov_mean


# -----------------------------------------------------------------------------
# Extended API: return covariance of **every** patch (no reduction)
# -----------------------------------------------------------------------------

def patchwise_covariance_all(feat: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """Compute covariance matrix for each patch individually.

    Args:
        feat: [B, C, H, W] (2-D) or 4-D reshaped from 3-D feature maps.
    Returns:
        Tensor of shape [B*L, C, C] where L is number of patches per image.
    """
    if feat.dim() != 4:
        raise ValueError("patchwise_covariance_all expects 4-D tensor [B,C,H,W]")

    B, C, H, W = feat.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h or pad_w:
        feat = F.pad(feat, (0, pad_w, 0, pad_h), mode='replicate')
        H, W = H + pad_h, W + pad_w

    unfold = F.unfold(feat, kernel_size=patch_size, stride=patch_size)  # [B, C*k*k, L]
    k2 = patch_size * patch_size
    L = unfold.size(-1)
    patches = unfold.view(B, C, k2, L).permute(0, 3, 2, 1).contiguous()  # [B, L, k2, C]
    patches = patches.view(-1, k2, C)  # [B*L, k2, C]

    cov_per_patch = _patch_covariance_batch(patches)  # [B*L, C, C]
    return cov_per_patch 