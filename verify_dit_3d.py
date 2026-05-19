import torch
from model.dit_modules.dit_3d import DiT_3D

def test_dit():
    print("[*] Initializing 3D DiT...")
    
    # Settings based on our data pipeline
    # Input size 32^3
    # Channels = 3 (1 Noisy Target + 1 B-Spline + 1 CT)
    # hidden_size=64 must be divisible by num_heads! (Using 4)
    model = DiT_3D(
        input_size=32, 
        patch_size=4, 
        in_channels=3, 
        hidden_size=64, 
        depth=2, 
        num_heads=4 
    )
    
    batch_size = 2
    # Create dummy input: [B, 3, 32, 32, 32]
    # 3 Channels: Residual, B-Spline Baseline, CT
    x = torch.randn(batch_size, 3, 32, 32, 32)
    
    # Timesteps
    t = torch.randint(0, 1000, (batch_size, 1)).float()
    
    print(f"    Input Shape: {x.shape}")
    
    output = model(x, t)
    
    print(f"    Output Shape: {output.shape}")
    
    # Expected output: [B, 1, 32, 32, 32] (Predicting residual only)
    if output.shape == (batch_size, 1, 32, 32, 32):
        print("✅ SUCCESS: 3D DiT Architecture is working.")
    else:
        print(f"❌ FAIL: Expected ({batch_size}, 1, 32, 32, 32), got {output.shape}")

if __name__ == "__main__":
    test_dit()