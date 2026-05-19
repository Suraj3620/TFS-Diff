import argparse
import torch
import numpy as np
import data as Data
import core.logger as Logger
# We don't strictly need the interpolator import here as it's inside the dataset class
# from core.bspline_interpolator_3d import BSplineInterpolator3D 

def main():
    # Load your config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_3d_unet_train.json')
    parser.add_argument('-p', '--phase', type=str, default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true') # Fixed: Added missing arg
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('--log_eval', action='store_true')
    parser.add_argument('--log_infer', action='store_true')
    
    args = parser.parse_args()
    
    # Parse Config
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # Force 3D Lazy Loader parameters manually to ensure robustness if config varies
    # We use the 'train' dataset logic
    dataset_opt = opt['datasets']['train']
    
    print("[*] Initializing Dataset...")
    dataset = Data.create_dataset(dataset_opt, 'train')
    
    print(f"[*] Dataset Size: {len(dataset)}")
    
    # We will inspect 50 random samples
    max_diffs = []
    mean_diffs = []
    
    print("[*] Sampling Data Differences (GT - Baseline)...")
    print("[*] Note: 'GT' is the raw high-res dose, 'Base' is the interpolated low-res.")
    
    # Iterate through a few samples
    num_samples = 50
    for i in range(num_samples):
        # Random sample
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]
        
        # In the dataset loader, we defined:
        # HR = Dose Norm [-1, 1]
        # LR/BASE = Baseline Norm [-1, 1]
        
        hr = data['HR'].squeeze().numpy()
        base = data['LR'].squeeze().numpy() 
        
        # Calculate Residual
        residual = hr - base
        
        # Stats
        max_val = np.max(np.abs(residual))
        mean_val = np.mean(np.abs(residual))
        
        max_diffs.append(max_val)
        mean_diffs.append(mean_val)
        
        if i < 5: # Print first 5 details
            print(f"Sample {idx}:")
            print(f"  GT Range:   [{hr.min():.4f}, {hr.max():.4f}]")
            print(f"  Base Range: [{base.min():.4f}, {base.max():.4f}]")
            print(f"  Max Diff:   {max_val:.6f}")
            print(f"  Mean Diff:  {mean_val:.6f}")

    avg_max = np.mean(max_diffs)
    avg_mean = np.mean(mean_diffs)
    
    print("\n" + "="*40)
    print(f"GLOBAL SIGNAL STRENGTH STATS (Over {num_samples} samples):")
    print(f"Average Max Error (Signal Peak): {avg_max:.6f}")
    print(f"Average Mean Error (Signal Mass): {avg_mean:.6f}")
    print("="*40)
    
    if avg_max < 0.1:
        print("CRITICAL DIAGNOSIS: The signal is too weak (< 0.1).")
        print(f"RECOMMENDATION: Apply SCALING factor of at least {1.0/avg_max:.1f}x.")
    else:
        print("Signal strength seems adequate (peaks > 0.1).")

if __name__ == "__main__":
    main()