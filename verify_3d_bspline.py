import numpy as np
import matplotlib.pyplot as plt
from core.bspline_interpolator_3d import BSplineInterpolator3D
import time

def main():
    print("[*] Testing 3D B-Spline Interpolation...")
    
    # 1. Create a dummy 3D volume (e.g., a sphere or Gaussian blob)
    D, H, W = 16, 16, 16
    x, y, z = np.ogrid[:D, :H, :W]
    center = (8, 8, 8)
    radius = 4
    # Create a binary sphere
    vol_lr = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2).astype(np.float32)
    
    # Add some gradients so it's not just binary
    vol_lr = scipy.ndimage.gaussian_filter(vol_lr, sigma=0.5)

    print(f"    Input Shape: {vol_lr.shape}")
    print(f"    Input Range: {vol_lr.min():.2f} to {vol_lr.max():.2f}")

    # 2. Initialize Interpolator
    # method='quintic' corresponds to 5-knot B-Spline
    try:
        bspline = BSplineInterpolator3D(method='quintic')
    except Exception as e:
        print(f"[ERROR] Scipy version might be too old for 'quintic'. Error: {e}")
        print("Falling back to 'linear' for test...")
        bspline = BSplineInterpolator3D(method='linear')

    # 3. Upscale to 32^3
    target_shape = (32, 32, 32)
    start = time.time()
    vol_hr = bspline.resize_3d(vol_lr, target_shape)
    end = time.time()

    print(f"    Output Shape: {vol_hr.shape}")
    print(f"    Output Range: {vol_hr.min():.2f} to {vol_hr.max():.2f}")
    print(f"    Time taken: {end - start:.4f}s")

    # 4. Visualize Central Slice
    slice_idx_lr = D // 2
    slice_idx_hr = target_shape[0] // 2

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(vol_lr[slice_idx_lr], cmap='hot', interpolation='nearest')
    ax[0].set_title(f"Input {D}x{D} (Slice {slice_idx_lr})")
    
    ax[1].imshow(vol_hr[slice_idx_hr], cmap='hot', interpolation='nearest')
    ax[1].set_title(f"B-Spline Output {target_shape[0]}x{target_shape[0]} (Slice {slice_idx_hr})")
    
    plt.savefig("verify_3d_bspline.png")
    print("[*] Saved visualization to verify_3d_bspline.png")

import scipy.ndimage # Import here for the dummy data creation
if __name__ == "__main__":
    main()