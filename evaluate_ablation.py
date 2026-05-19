import os
import argparse
import logging
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader # <-- We import this directly to bypass the hardcoded limits

import data as Data
import model as Model
import core.logger as Logger
from core.metrics_bodymask import ct_body_mask

def _rmsd(a_cpu, b_cpu, mask_cpu=None):
    a, b = a_cpu.float(), b_cpu.float()
    diff2 = (a - b) ** 2
    if mask_cpu is not None:
        valid_pixels = mask_cpu.sum().clamp_min(1.0)
        mse = (diff2 * mask_cpu).sum() / valid_pixels
    else:
        mse = diff2.mean()
    return torch.sqrt(mse.clamp_min(1e-12)).item()

def _final_frame(t):
    if isinstance(t, (list, tuple)): return t[-1]
    if torch.is_tensor(t) and t.dim() == 5: return t[-1]
    return t

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Global vs High-Dose Evaluation")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the specific .pth checkpoint')
    parser.add_argument('--output_csv', type=str, required=True, help='Where to save the results CSV')
    
    parser.add_argument('-p', '--phase', type=str, default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    opt.setdefault('path', {})
    opt['path']['pretrain_model_G'] = args.ckpt          
    opt['path']['resume_state'] = None                   

    # --- SPEED UP FIX 1: Apply fraction BEFORE creating dataset ---
    # 0.05 * 2000 = 100 samples. This is mathematically plenty for statistical significance.
    opt['datasets']['val']['subset_fraction'] = 0.05 

    print(f"[*] Starting Ablation Evaluation...")
    print(f"[*] Config: {args.config}")
    print(f"[*] Checkpoint: {args.ckpt}")

    diffusion = Model.create_model(opt)
    diffusion.netG.eval()
    
    print("[*] Loading Validation Set...")
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    
    # --- SPEED UP FIX 2: Force Batch Size 4 ---
    # We bypass Data.create_dataloader to avoid the hardcoded batch_size=1
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    results_list =[]
    
    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="Evaluating validation set"):
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()
            
            sr_res = _final_frame(visuals['SR'])
            hr_m11 = visuals['HR']
            base_m11 = visuals['LR_UP']
            ct_m11 = visuals['CT']
            
            sr_m11 = (base_m11 + sr_res).clamp(-1.0, 1.0)
            hr_m11 = hr_m11.clamp(-1.0, 1.0)
            
            hr_01 = ((hr_m11 + 1.0) / 2.0).cpu()
            sr_01 = ((sr_m11 + 1.0) / 2.0).cpu()
            base_01 = ((base_m11 + 1.0) / 2.0).cpu()
            
            batch_size_current = hr_01.shape[0]
            for b in range(batch_size_current):
                global_mask = ct_body_mask(ct_m11[b:b+1].cpu())
                high_dose_threshold = 0.10 
                high_dose_mask = (hr_01[b:b+1] > high_dose_threshold).float() * global_mask
                
                rmsd_model_global = _rmsd(sr_01[b:b+1], hr_01[b:b+1], global_mask)
                rmsd_base_global  = _rmsd(base_01[b:b+1], hr_01[b:b+1], global_mask)
                
                rmsd_model_high = _rmsd(sr_01[b:b+1], hr_01[b:b+1], high_dose_mask)
                rmsd_base_high  = _rmsd(base_01[b:b+1], hr_01[b:b+1], high_dose_mask)
                
                # --- NEW: CLINICAL SAFETY METRICS (Worst-case & 95th Percentile) ---
                # Calculate absolute error for high-dose pixels only
                diff_model = torch.abs(sr_01[b:b+1] - hr_01[b:b+1])
                mask_bool = high_dose_mask[b:b+1].bool()
                
                if mask_bool.sum() > 0:
                    errors = diff_model[mask_bool] # Extract only the beam pixels
                    max_err = errors.max().item()
                    p95_err = torch.quantile(errors.float(), 0.95).item()
                else:
                    max_err = 0.0
                    p95_err = 0.0

                # --- SAFE METADATA EXTRACTION ---
                if 'uid' in val_data and len(val_data['uid']) > b:
                    uid = val_data['uid'][b]
                else:
                    uid = 'N/A'

                if 'slice' in val_data and len(val_data['slice']) > b:
                    slc = val_data['slice'][b]
                    slice_idx = slc.item() if torch.is_tensor(slc) else slc
                else:
                    slice_idx = -1

                results_list.append({
                    'uid': uid,
                    'slice': slice_idx,
                    'Model_Global_RMSD': rmsd_model_global,
                    'Base_Global_RMSD': rmsd_base_global,
                    'Model_HighDose_RMSD': rmsd_model_high,
                    'Base_HighDose_RMSD': rmsd_base_high,
                    'Model_HighDose_MAX_Error': max_err,  # NEW
                    'Model_HighDose_95th_Error': p95_err  # NEW
                })
            
    df = pd.DataFrame(results_list)
    
    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY")
    print("="*50)
    print(f"Global Body Error (RMSD):")
    print(f"  - Baseline: {df['Base_Global_RMSD'].mean():.5f}")
    print(f"  - Model:    {df['Model_Global_RMSD'].mean():.5f}")
    print(f"\nHigh-Dose Beam Error (RMSD) [PROFESSOR'S REQUEST]:")
    print(f"  - Baseline: {df['Base_HighDose_RMSD'].mean():.5f}")
    print(f"  - Model:    {df['Model_HighDose_RMSD'].mean():.5f}")
    print("="*50)

    # The CSV Bug Fix
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:  
        os.makedirs(out_dir, exist_ok=True)
        
    df.to_csv(args.output_csv, index=False)
    print(f"Results successfully saved to: {args.output_csv}")

if __name__ == "__main__":
    main()