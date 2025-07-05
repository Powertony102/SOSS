#!/usr/bin/env python3
"""
Test script to demonstrate the fix for prototype loss being zero
"""

import torch
import logging
from myutils.prototype_separation_fixed import PrototypeMemoryFixed

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_prototype_fix():
    """Test the fixed prototype memory module"""
    print("="*60)
    print("Testing Fixed PrototypeMemory Module")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create fixed prototype memory with adaptive threshold
    proto_mem = PrototypeMemoryFixed(
        num_classes=1,  # LA dataset: 1 foreground class
        feat_dim=64,
        proto_momentum=0.9,
        conf_thresh=0.3,  # Start with lower threshold
        conf_thresh_max=0.85,  # Gradually increase to this
        conf_thresh_rampup=1000,  # Over 1000 iterations
        lambda_intra=0.3,
        lambda_inter=0.1,
        margin_m=1.5,
        min_pixels_per_class=5,  # Ensure at least 5 pixels per class
        use_labeled_fallback=True,  # Use labeled pixels as fallback
        device=device
    ).to(device)
    
    print(f"Created PrototypeMemoryFixed with adaptive threshold")
    
    # Simulate early training with poor predictions
    batch_size = 2
    H, W, D = 32, 32, 16
    
    for iteration in range(0, 1100, 100):
        print(f"\n--- Iteration {iteration} ---")
        
        # Create test data simulating poor early predictions
        feat = torch.randn(batch_size, 64, H, W, D, device=device, requires_grad=True)
        
        # Simulate poor predictions (low confidence)
        if iteration < 500:
            # Early training: very poor predictions
            logits = torch.randn(batch_size, 2, H, W, D, device=device) * 0.5
        else:
            # Later training: better predictions
            logits = torch.randn(batch_size, 2, H, W, D, device=device) * 2.0
        
        pred = torch.softmax(logits, dim=1)
        label = torch.randint(0, 2, (batch_size, 1, H, W, D), device=device)
        is_labelled = torch.tensor([True, False], device=device)
        
        # Compute losses
        losses = proto_mem(
            feat=feat,
            label=label,
            pred=pred,
            is_labelled=is_labelled,
            epoch_idx=iteration
        )
        
        print(f"  Confidence threshold: {losses['current_conf_thresh']:.3f}")
        print(f"  Max confidence: {losses['max_confidence']:.3f}")
        print(f"  Mean confidence: {losses['mean_confidence']:.3f}")
        print(f"  Confident pixels: {losses['n_confident_pixels']}")
        print(f"  Initialized prototypes: {losses['n_initialized_protos']}")
        print(f"  Intra loss: {losses['intra'].item():.6f}")
        print(f"  Inter loss: {losses['inter'].item():.6f}")
        print(f"  Total loss: {losses['total'].item():.6f}")
        
        # Test backward pass
        if losses['total'].item() > 0:
            losses['total'].backward()
            print(f"  ✓ Backward pass successful")
            if feat.grad is not None:
                print(f"  ✓ Gradients computed: {feat.grad.norm().item():.6f}")
        else:
            print(f"  ⚠ Loss is zero - this should be rare with the fix")

if __name__ == "__main__":
    test_prototype_fix()