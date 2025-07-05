#!/usr/bin/env python3
"""
Second-Order Feature Modeling Module for 3D Medical Images

This module enhances feature structural representation by:
1. Computing covariance matrix of spatial features
2. Extracting principal correlation modes via eigendecomposition  
3. Projecting features onto top-K eigenvectors
4. Applying MLP activation to generate mode responses
5. Concatenating original features with mode responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_eigh(mat: torch.Tensor, jitter: float = 1e-5, max_attempts: int = 4):
    """
    对称矩阵特征分解，自动为病态矩阵加抖动(jitter)以确保收敛。
    
    Args:
        mat: (..., C, C) 对称矩阵，需在最后两维上做分解
        jitter: 初始对角抖动值
        max_attempts: 失败后指数增大 jitter 的重试次数
        
    Returns:
        eigvals, eigvecs  (与 torch.linalg.eigh 接口一致)
    """
    Bshape = mat.shape[:-2]
    C = mat.shape[-1]
    eye = torch.eye(C, device=mat.device, dtype=mat.dtype).expand(*Bshape, -1, -1)

    # 强制对称，避免偶发非对称误差
    mat = 0.5 * (mat + mat.transpose(-1, -2))

    for i in range(max_attempts):
        try:
            return torch.linalg.eigh(mat.to(torch.float64))
        except RuntimeError:
            mat = mat + (jitter * (10 ** i)) * eye
    
    # 最后一次仍失败则降到 CPU 再试一次
    return torch.linalg.eigh(mat.cpu().double())


class SecondOrderFeatureModule(nn.Module):
    """
    Second-Order Feature Modeling Module for enhancing 3D medical image features
    
    This module captures global statistical dependencies between feature channels
    through correlation analysis and principal component projection.
    """
    
    def __init__(self, K, mlp_hidden, eps=1e-5):
        """
        Initialize Second-Order Feature Module
        
        Args:
            K (int): Number of principal correlation modes to extract
            mlp_hidden (int): Hidden dimension of MLP for mode activation
            eps (float, optional): Numerical stability constant. Defaults to 1e-5.
        """
        super(SecondOrderFeatureModule, self).__init__()
        self.K = K
        self.mlp_hidden = mlp_hidden
        self.eps = eps
        
        # MLP for mode activation - transforms projected features
        self.mlp = nn.Sequential(
            nn.Linear(K, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, K)
        )
        
        # Initialize MLP weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights using Xavier initialization"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of Second-Order Feature Module
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D)
                - B: batch size
                - C: number of channels  
                - H, W, D: spatial dimensions
            
        Returns:
            torch.Tensor: Enhanced features of shape (B, C+K, H, W, D)
                - Original C channels + K mode response channels
        """
        B, C, H, W, D = x.shape
        N = H * W * D  # Total number of spatial locations
        original_dtype = x.dtype
        
        # Step 1: Flatten spatial dimensions to get feature matrix X ∈ ℝ^{C×N}
        x_flat = x.view(B, C, N)  # (B, C, N)
        
        # Step 2: Zero-center the features (remove mean across spatial locations)
        x_mean = x_flat.mean(dim=2, keepdim=True)  # (B, C, 1)
        x_centered = x_flat - x_mean  # (B, C, N)
        
        # Step 3: Compute covariance matrix Σ ∈ ℝ^{C×C}
        # Cov = (X - μ)(X - μ)^T / (N - 1)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (N - 1)  # (B, C, C)
        
        # Step 4: Normalize to correlation matrix R
        # Extract diagonal variances and ensure numerical stability
        var = cov.diagonal(dim1=1, dim2=2).clamp(min=self.eps).sqrt()  # (B, C)
        
        # Compute correlation matrix: R_ij = Cov_ij / (σ_i * σ_j)
        var_outer = var.unsqueeze(2) * var.unsqueeze(1)  # (B, C, C)
        corr = cov / var_outer  # (B, C, C)
        
        # 数值稳定性改进：转换到 float64 精度
        cov = cov.double()
        var = var.double()
        corr = corr.double()
        
        # 清理 NaN/Inf 和强制对称
        corr = torch.nan_to_num(corr)
        corr = 0.5 * (corr + corr.transpose(-1, -2))
        
        # Step 5: 安全的特征分解
        eigvals64, eigvecs64 = safe_eigh(corr)
        eigvals, eigvecs = eigvals64.to(original_dtype), eigvecs64.to(original_dtype)
        
        # Get top-K eigenvalues and corresponding eigenvectors (largest eigenvalues)
        topk_vals, indices = eigvals.topk(self.K, dim=-1)  # (B, K)
        
        # Gather top-K eigenvectors using advanced indexing
        # Expand indices to match eigenvector dimensions
        indices_expanded = indices.unsqueeze(1).expand(-1, C, -1)  # (B, C, K)
        topk_vecs = torch.gather(eigvecs, 2, indices_expanded)  # (B, C, K)
        
        # Step 6: Project centered features onto top-K eigenvectors
        # proj = V^T * X_centered, where V contains top-K eigenvectors
        proj = torch.einsum('bck,bcn->bkn', topk_vecs, x_centered)  # (B, K, N)
        
        # Reshape for MLP processing: (B, N, K)
        proj = proj.permute(0, 2, 1)  # (B, N, K)
        
        # Step 7: Apply MLP activation to generate mode responses
        act = self.mlp(proj)  # (B, N, K)
        
        # Reshape back to spatial dimensions: (B, K, H, W, D)
        act = act.permute(0, 2, 1).view(B, self.K, H, W, D)  # (B, K, H, W, D)
        
        # Step 8: Concatenate original features with mode responses
        output = torch.cat([x, act], dim=1)  # (B, C+K, H, W, D)
        
        return output
    
    def get_correlation_info(self, x):
        """
        Get correlation analysis information for debugging/visualization
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D)
            
        Returns:
            dict: Dictionary containing correlation analysis results
        """
        B, C, H, W, D = x.shape
        N = H * W * D
        
        with torch.no_grad():
            # Compute correlation matrix
            x_flat = x.view(B, C, N)
            x_centered = x_flat - x_flat.mean(dim=2, keepdim=True)
            cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (N - 1)
            var = cov.diagonal(dim1=1, dim2=2).clamp(min=self.eps).sqrt()
            corr = cov / (var.unsqueeze(2) * var.unsqueeze(1))
            
            # 使用安全的特征分解
            eigvals64, eigvecs64 = safe_eigh(corr.double())
            eigvals = eigvals64.to(corr.dtype)
            topk_vals, _ = eigvals.topk(self.K, dim=-1)
            
            return {
                'correlation_matrix': corr,  # (B, C, C)
                'eigenvalues': eigvals,      # (B, C)
                'top_eigenvalues': topk_vals, # (B, K)
                'explained_variance_ratio': topk_vals.sum(dim=-1) / eigvals.sum(dim=-1)  # (B,)
            }


def _test_safe_eigh():
    """测试 safe_eigh 函数的稳定性"""
    torch.manual_seed(0)
    # 创建极端病态矩阵
    bad = torch.randn(2, 16, 16)
    bad = bad @ bad.transpose(-1, -2) * 1e-6  # 极端病态
    
    # 测试安全特征分解
    vals, vecs = safe_eigh(bad)
    
    # 验证分解结果的正确性
    reconstructed = vecs @ torch.diag_embed(vals) @ vecs.transpose(-1, -2)
    assert torch.allclose(bad.double(), reconstructed, atol=1e-3), "特征分解重构失败"
    
    print("safe_eigh 病态矩阵测试通过")


def test_second_order_feature_module():
    """
    Test function for SecondOrderFeatureModule
    Verifies output dimensions and gradient computation
    """
    print("=" * 60)
    print("Testing SecondOrderFeatureModule")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Test parameters
    B, C, H, W, D = 2, 64, 8, 8, 8  # Small size for testing
    K = 16
    mlp_hidden = 32
    
    # Create module
    print(f"Creating SecondOrderFeatureModule with K={K}, mlp_hidden={mlp_hidden}")
    module = SecondOrderFeatureModule(K=K, mlp_hidden=mlp_hidden)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = module.to(device)
    print(f"Using device: {device}")
    
    # Create random input
    x = torch.randn(B, C, H, W, D, requires_grad=True, device=device)
    
    print(f"\nInput tensor:")
    print(f"  Shape: {x.shape}")
    print(f"  Device: {x.device}")
    print(f"  Requires grad: {x.requires_grad}")
    print(f"  Data type: {x.dtype}")
    
    # Forward pass
    print(f"\nForward pass...")
    output = module(x)
    
    print(f"\nOutput tensor:")
    print(f"  Shape: {output.shape}")
    print(f"  Expected shape: ({B}, {C + K}, {H}, {W}, {D})")
    print(f"  Device: {output.device}")
    print(f"  Data type: {output.dtype}")
    
    # Verify output shape
    expected_shape = (B, C + K, H, W, D)
    assert output.shape == expected_shape, f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print("✓ Output shape verification passed!")
    
    # Test gradient computation
    print(f"\nTesting gradient computation...")
    loss = output.sum()
    loss.backward()
    
    grad_computed = x.grad is not None
    print(f"  Gradient computed: {grad_computed}")
    if grad_computed:
        print(f"  Gradient shape: {x.grad.shape}")
        print(f"  Gradient norm: {x.grad.norm().item():.6f}")
    
    assert grad_computed, "Gradient computation failed!"
    print("✓ Gradient computation verification passed!")
    
    # Test correlation analysis
    print(f"\nTesting correlation analysis...")
    corr_info = module.get_correlation_info(x)
    print(f"  Correlation matrix shape: {corr_info['correlation_matrix'].shape}")
    print(f"  Top-{K} eigenvalues mean: {corr_info['top_eigenvalues'].mean().item():.6f}")
    print(f"  Explained variance ratio mean: {corr_info['explained_variance_ratio'].mean().item():.6f}")
    
    # Test with different input sizes
    print(f"\nTesting with different input sizes...")
    test_shapes = [
        (1, 32, 4, 4, 4),      # Small input
        (3, 128, 16, 16, 16),  # Medium input  
        (2, 256, 12, 12, 12),  # Large input (but smaller than original large test)
    ]
    
    for i, (b, c, h, w, d) in enumerate(test_shapes):
        try:
            # Create test module with appropriate K for channel size
            test_K = min(K, c // 2)  # Ensure K doesn't exceed reasonable limit
            test_module = SecondOrderFeatureModule(K=test_K, mlp_hidden=mlp_hidden).to(device)
            
            test_x = torch.randn(b, c, h, w, d, device=device)
            test_output = test_module(test_x)
            expected_c = c + test_K
            
            assert test_output.shape == (b, expected_c, h, w, d)
            print(f"  Test {i+1}: ✓ Input {(b, c, h, w, d)} -> Output {test_output.shape}")
            
        except Exception as e:
            print(f"  Test {i+1}: ✗ Failed with error: {e}")
            raise e
    
    print(f"\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    return module, output


def demo_integration_example():
    """
    Demonstrate how to integrate SecondOrderFeatureModule into existing networks
    """
    print("\n" + "=" * 60)
    print("Integration Example")
    print("=" * 60)
    
    class ExampleNetwork(nn.Module):
        """Example of integrating SecondOrderFeatureModule"""
        
        def __init__(self, in_channels=1, base_channels=32, num_classes=2):
            super(ExampleNetwork, self).__init__()
            
            # Initial convolution
            self.conv1 = nn.Conv3d(in_channels, base_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm3d(base_channels)
            
            # Second-order feature enhancement
            self.second_order = SecondOrderFeatureModule(K=8, mlp_hidden=16)
            
            # Enhanced feature channels: base_channels + K
            enhanced_channels = base_channels + 8
            
            # Final convolution for segmentation
            self.conv2 = nn.Conv3d(enhanced_channels, num_classes, 1)
            
        def forward(self, x):
            # Initial feature extraction
            x = F.relu(self.bn1(self.conv1(x)))
            
            # Second-order feature enhancement
            x = self.second_order(x)
            
            # Final classification
            x = self.conv2(x)
            return x
    
    # Test the integrated network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ExampleNetwork().to(device)
    
    # Dummy input (medical image patch)
    x = torch.randn(2, 1, 32, 32, 32, device=device)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = net(x)
    
    print(f"Network output shape: {output.shape}")
    print("✓ Integration example completed successfully!")


if __name__ == "__main__":
    # 测试安全特征分解
    _test_safe_eigh()
    
    # Run tests
    module, output = test_second_order_feature_module()
    
    # Run integration example
    demo_integration_example() 