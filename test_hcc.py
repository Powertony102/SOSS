import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from myutils.hcc_loss import hierarchical_coral, parse_hcc_weights


def test_hierarchical_coral():
    """Test hierarchical coral loss with random features"""
    print("Testing Hierarchical Covariance Consistency Loss...")
    
    # Test parameters
    batch_size = 2
    num_layers = 5
    weights = [0.5, 0.5, 1.0, 1.0, 1.5]
    
    # Create random features for 5 encoder layers (simulating VNet encoder output)
    features_s = []
    features_t = []
    
    # Layer dimensions: progressively smaller spatial size, more channels
    layer_configs = [
        (16, 56, 56, 40),   # Layer 1: [B, 16, 56, 56, 40]
        (32, 28, 28, 20),   # Layer 2: [B, 32, 28, 28, 20]
        (64, 14, 14, 10),   # Layer 3: [B, 64, 14, 14, 10]
        (128, 7, 7, 5),     # Layer 4: [B, 128, 7, 7, 5]
        (256, 4, 4, 3),     # Layer 5: [B, 256, 4, 4, 3]
    ]
    
    for i, (c, h, w, d) in enumerate(layer_configs):
        # Create random features for student and teacher networks
        f_s = torch.randn(batch_size, c, d, h, w, requires_grad=True)
        f_t = torch.randn(batch_size, c, d, h, w, requires_grad=True)
        features_s.append(f_s)
        features_t.append(f_t)
        print(f"Layer {i+1}: {f_s.shape}")
    
    # Test patch mode
    print("\nTesting patch mode...")
    loss_patch = hierarchical_coral(features_s, features_t, weights, 
                                  cov_mode='patch', patch_size=4)
    print(f"HCC Loss (patch mode): {loss_patch.item():.6f}")
    
    # Test full mode
    print("\nTesting full mode...")
    loss_full = hierarchical_coral(features_s, features_t, weights, 
                                 cov_mode='full', patch_size=4)
    print(f"HCC Loss (full mode): {loss_full.item():.6f}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    loss_patch.backward()
    print("Backward pass successful!")
    
    # Test that loss is non-negative
    assert loss_patch.item() >= 0, "Loss should be non-negative"
    assert loss_full.item() >= 0, "Loss should be non-negative"
    
    print("✓ All tests passed!")


def test_parse_hcc_weights():
    """Test HCC weights parsing"""
    print("\nTesting HCC weights parsing...")
    
    # Test valid input
    weights_str = "0.5,0.5,1,1,1.5"
    weights = parse_hcc_weights(weights_str, num_layers=5)
    expected = [0.5, 0.5, 1.0, 1.0, 1.5]
    assert weights == expected, f"Expected {expected}, got {weights}"
    print(f"✓ Parsed weights: {weights}")
    
    # Test invalid length
    try:
        parse_hcc_weights("0.5,1.0", num_layers=5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test invalid format
    try:
        parse_hcc_weights("0.5,abc,1.0", num_layers=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("✓ All parsing tests passed!")


if __name__ == "__main__":
    test_hierarchical_coral()
    test_parse_hcc_weights()
    print("\n🎉 All tests completed successfully!") 