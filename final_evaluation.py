import os
import argparse
import logging
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

import data as Data
import model as Model
import core.logger as Logger
from core.metrics_bodymask import ct_body_mask

# --- METRIC HELPERS ---
def _rmsd(a_cpu, b_cpu, mask_cpu=None):
    a, b = a_cpu.float(), b_cpu.float()
    diff2 = (a - b) ** 2
    if mask_cpu is not None:
        mse = (diff2 * mask_cpu).sum() / mask_cpu.sum().clamp_min(1.0)
    else:
        mse = diff2.mean()
    return torch.sqrt(mse.clamp_min(1e-12)).item()

def _psnr(a_cpu, b_cpu, mask_cpu=None):
    a, b = a_cpu.float(), b_cpu.float()
    diff2 = (a - b) ** 2
    if mask_cpu is not None:
        mse = (diff2 * mask_cpu).sum() / mask_cpu.sum().clamp_min(1.0)
    else:
        mse = diff2.mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-12)).item()

def _final_frame(t):
    if isinstance(t, (list, tuple)): return t[-1]
    if torch.is_tensor(t) and t.dim() == 5: return t[-1]
    return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final evaluation script with CPU-based metrics.")
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_train.json', help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None, help="Comma-separated GPU ids")
    parser.add_argument('--output_csv', type=str, default='experiments/final_evaluation_results_CORRECTED.csv', help="Path to save the detailed results")

    # logger flags
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # checkpoint argument
    parser.add_argument('--ckpt', type=str, default=None, help='Path to *_gen.pth to evaluate')

    args = parser.parse_args()
    args.phase = 'val'

    # Build opt from args first
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # If --ckpt is provided, force it into opt['path'] and clear resume_state
    if args.ckpt:
        opt.setdefault('path', {})
        opt['path']['pretrain_model_G'] = args.ckpt          
        opt['path']['resume_state'] = None                   
        print("[sanity] using ckpt:", opt['path']['pretrain_model_G'])
    else:
        print("[sanity] pretrain_model_G from opt:", opt['path'].get('pretrain_model_G'))
        print("[sanity] resume_state from opt:", opt['path'].get('resume_state'))

    # Normal logger setup
    Logger.setup_logger(None, opt['path']['log'], 'final_eval_cpu', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info("Starting Final Evaluation with CPU-based Metric Calculation...")
    # Create the model and load weights
    diffusion = Model.create_model(opt)
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    diffusion.netG.eval()

    # LPIPS handle
    lpips_model = diffusion.netG.module.lpips_loss if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG.lpips_loss
    if lpips_model: lpips_model.eval()

    results_list = []
    
    # --- MAIN EVALUATION LOOP ---
    for val_data in tqdm(val_loader, desc="Evaluating validation set"):
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()
        
        sr_final_gpu = _final_frame(visuals['SR'])
        hr_m11_gpu = visuals['HR']
        lr_up_m11_gpu = visuals['LR_UP']
        ct_m11_gpu = visuals['CT']
        
        sr_m11_clamped = sr_final_gpu.clamp(-1.0, 1.0)
        
        hr_01_gpu = (hr_m11_gpu + 1) / 2
        sr_01_gpu = (sr_m11_clamped + 1) / 2
        lr_up_01_gpu = (lr_up_m11_gpu + 1) / 2
        
        # --- Move all tensors for PSNR/RMSD to CPU for calculation ---
        hr_01_cpu = hr_01_gpu.cpu()
        sr_01_cpu = sr_01_gpu.cpu()
        lr_up_01_cpu = lr_up_01_gpu.cpu()
        mask_cpu = ct_body_mask(ct_m11_gpu.cpu())
        
        # --- Calculate Metrics (on CPU) ---
        rmsd_model = _rmsd(sr_01_cpu, hr_01_cpu, mask_cpu)
        psnr_model = _psnr(sr_01_cpu, hr_01_cpu, mask_cpu)
        
        rmsd_baseline = _rmsd(lr_up_01_cpu, hr_01_cpu, mask_cpu)
        psnr_baseline = _psnr(lr_up_01_cpu, hr_01_cpu, mask_cpu)
        
        # LPIPS is safe to run on the GPU
        lpips_model_val = lpips_model(sr_m11_clamped, hr_m11_gpu).mean().item() if lpips_model else -1.0
        lpips_baseline = lpips_model(lr_up_m11_gpu, hr_m11_gpu).mean().item() if lpips_model else -1.0
        
        results_list.append({
            'uid': val_data.get('uid', ['N/A'])[0], 'slice': val_data.get('slice', [-1])[0].item(),
            'psnr_model': psnr_model, 'rmsd_model': rmsd_model, 'lpips_model': lpips_model_val,
            'psnr_baseline': psnr_baseline, 'rmsd_baseline': rmsd_baseline, 'lpips_baseline': lpips_baseline
        })
        
    # Save detailed results to CSV
    results_df = pd.DataFrame(results_list)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    logger.info(f"Detailed CORRECTED evaluation results saved to: {args.output_csv}")
    
    logger.info("\n--- Final Head-to-Head Performance (Mean ± Std Dev) ---")
    summary_data = {}
    for method in ['model', 'baseline']:
        summary_data[method] = {
            'PSNR':  f"{results_df[f'psnr_{method}'].mean():.2f} ± {results_df[f'psnr_{method}'].std():.2f}",
            'RMSD':  f"{results_df[f'rmsd_{method}'].mean()*100:.2f}% ± {results_df[f'rmsd_{method}'].std()*100:.2f}%",
            'LPIPS': f"{results_df[f'lpips_{method}'].mean():.3f} ± {results_df[f'lpips_{method}'].std():.3f}",
        }
    summary_df = pd.DataFrame(summary_data).rename(columns={'model': 'TFSDiff (Ours)', 'baseline': 'Bilinear Baseline'})
    logger.info("\n" + summary_df.to_string())
    
    logger.info("\n--- Top 5 Worst Performing Samples for the Model (by RMSD) ---")
    logger.info("\n" + results_df.sort_values(by='rmsd_model', ascending=False).head(5).to_string())
    
    logger.info("\nFinal evaluation complete.")