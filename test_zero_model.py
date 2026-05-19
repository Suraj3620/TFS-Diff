import argparse
import torch
import numpy as np
import data as Data
import core.logger as Logger
from core.metrics_bodymask import ct_body_mask

def _rmsd(a, b, mask=None):
    a, b = a.float(), b.float()
    diff2 = (a - b) ** 2
    if mask is not None:
        mse = (diff2 * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        mse = diff2.mean()
    return torch.sqrt(mse.clamp_min(1e-12)).item()

def main():
    # 1. Setup Config (Reusing your existing config structure)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_train.json')
    parser.add_argument('-p', '--phase', type=str, default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # 2. Load Validation Data
    print("[*] Loading Validation Set...")
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    
    rmsd_list = []
    
    print("[*] Running Zero-Weight Simulation (Model Output = 0)...")
    
    for i, val_data in enumerate(val_loader):
        # Unpack Data
        # SR in your dataset dict is [Baseline, CT]
        baseline_norm = val_data['SR'][:, 0:1, :, :, :] 
        ct_img = val_data['SR'][:, 1:2, :, :, :]
        hr_gt = val_data['HR']
        
        # --- SIMULATE ZERO WEIGHTS ---
        # Instead of running the model, we force the residual to be 0
        pred_residual = torch.zeros_like(hr_gt)
        
        # Reconstruct (Identity Mapping)
        sr_reconstructed = baseline_norm + pred_residual
        
        # Metrics
        sr_clamped = sr_reconstructed.clamp(-1.0, 1.0)
        hr_clamped = hr_gt.clamp(-1.0, 1.0)
        mask = ct_body_mask(ct_img)
        
        val_rmsd = _rmsd(sr_clamped, hr_clamped, mask)
        rmsd_list.append(val_rmsd)
        
        if i < 3:
            print(f"Sample {i}: Zero-Model RMSD = {val_rmsd:.6f}")

    avg_rmsd = np.mean(rmsd_list)
    print("="*40)
    print(f"RESULTS FOR 'ZERO WEIGHT' HYPOTHESIS:")
    print(f"Average Validation RMSD: {avg_rmsd:.6f}")
    print("="*40)
    print("Compare this number to your trained model's validation loss.")
    print("If they are similar, your model has learned NOTHING.")

if __name__ == "__main__":
    main()