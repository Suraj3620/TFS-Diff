import torch
import numpy as np
import matplotlib.pyplot as plt
from data.LRHR_Fusion_dataset import LRHRDataset

# Fake opt
opt = {
    'dataroot': 'data/CT_Dose_Dataset/train', 
    'l_resolution': 25, 'r_resolution': 100,
    'use_shuffle': False
}

try:
    ds = LRHRDataset(opt, phase='val')
    item = ds[0]

    hr = item['HR'].squeeze().numpy()
    lr_up = item['LR'].squeeze().numpy()
    res = item['RES'].squeeze().numpy()

    print(f"HR Range: {hr.min():.3f} to {hr.max():.3f}")
    print(f"LR Range: {lr_up.min():.3f} to {lr_up.max():.3f}")
    print(f"RES Range: {res.min():.3f} to {res.max():.3f}")
    
    mse = np.mean((hr - lr_up)**2)
    print(f"Baseline MSE: {mse:.6f}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(hr, cmap='gray'); ax[0].set_title('HR (Ground Truth)')
    ax[1].imshow(lr_up, cmap='gray'); ax[1].set_title('B-Spline Upsampled')
    ax[2].imshow(res, cmap='bwr'); ax[2].set_title('Residual (Target)')
    plt.show()
    plt.savefig("bspline_verification.png")
    print("[*] Saved bspline_verification.png")

except Exception as e:
    print(f"Error: {e}")
    print("Did you run the indexer/splitter scripts first?")