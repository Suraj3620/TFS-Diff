import wandb
import argparse
import torch
import numpy as np
import data as Data
import model as Model
from core.metrics_bodymask import ct_body_mask
from data.CT_Dose_Dataset_3D_Lazy import CT_Dose_Dataset_3D_Lazy

def _final_frame(t):
    if isinstance(t, (list, tuple)): return t[-1]
    if torch.is_tensor(t) and t.dim() == 6: return t[-1]
    return t

def _rmsd(a, b, mask=None):
    a, b = a.float(), b.float()
    diff2 = (a - b) ** 2
    if mask is not None:
        mse = (diff2 * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        mse = diff2.mean()
    return torch.sqrt(mse.clamp_min(1e-12)).item()

def train():
    wandb.init()
    config = wandb.config

    opt = {
        'name': 'Sweep_Run',
        'phase': 'train',
        'gpu_ids': [0],
        'distributed': False,
        'path': {
            'log': 'experiments/sweep/logs',
            'results': 'experiments/sweep/results',
            'checkpoint': 'experiments/sweep/ckpt',
            'resume_state': None,
            'pretrain_model_G': None
        },
        'datasets': {
            'train': {
                'name': 'CT_Dose_Dataset_3D_Lazy',
                'dataroot': 'data/CT_Dose_Dataset/train/index.csv',
                'patch_size': 32,
                'input_size': 16,
                'batch_size': config.batch_size,
                'num_workers': 4,
                'use_shuffle': True,
                'subset_fraction': 0.1  # 10% of 20k = 2000 samples (Good for training)
            },
            'val': {
                'name': 'CT_Dose_Dataset_3D_Lazy',
                'dataroot': 'data/CT_Dose_Dataset/val/index.csv',
                'patch_size': 32,
                'input_size': 16,
                'batch_size': 1,
                'num_workers': 2,
                # CRITICAL CHANGE: Use only 1% of validation data (approx 20 samples)
                # This makes validation take seconds, not hours.
                'subset_fraction': 0.01 
            }
        },
        'model': {
            'which_model_G': 'dit_3d',
            'dit': {
                'patch_size': config.internal_patch_size,
                'hidden_size': config.hidden_size,
                'depth': config.depth,
                'num_heads': config.num_heads,
                'mlp_ratio': 4.0
            },
            'beta_schedule': {
                'train': { 'schedule': 'linear', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 2e-2 },
                'val':   { 'schedule': 'linear', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 2e-2 }
            },
            'diffusion': { 'image_size': 32, 'channels': 1, 'conditional': True }
        },
        'train': {
            'n_iter': 4000, # 2 "Sweep Epochs" (2000 steps * 2)
            'val_freq': 500, # Validate frequently
            'save_checkpoint_freq': 10000,
            'optimizer': { 'type': 'adamw', 'lr': config.learning_rate, 'weight_decay': 0.0 },
            'lr_scheduler': { 'type': 'cosine_annealing', 'eta_min': 1e-7 },
            'loss_type': 'l1',
            'lambda_img': 0.0, 'use_ssim': False, 'lambda_lpips': 0.0
        }
    }

    try:
        # Initialize with Subsampling Parameters
        train_set = CT_Dose_Dataset_3D_Lazy(
            csv_path=opt['datasets']['train']['dataroot'],
            patch_size=opt['datasets']['train']['patch_size'],
            input_size=opt['datasets']['train']['input_size'],
            phase='train',
            subset_fraction=opt['datasets']['train']['subset_fraction']
        )
        
        val_set = CT_Dose_Dataset_3D_Lazy(
            csv_path=opt['datasets']['val']['dataroot'],
            patch_size=opt['datasets']['val']['patch_size'],
            input_size=opt['datasets']['val']['input_size'],
            phase='val',
            subset_fraction=opt['datasets']['val']['subset_fraction']
        )

        train_loader = Data.DataLoader(train_set, batch_size=opt['datasets']['train']['batch_size'], 
                                     shuffle=True, num_workers=4, pin_memory=True)
        # Validation loader
        val_loader = Data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

        diffusion = Model.create_model(opt)
        diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
    except Exception as e:
        print(f"Setup failed: {e}")
        return

    current_step = 0
    while current_step < opt['train']['n_iter']:
        for train_data in train_loader:
            current_step += 1
            if current_step > opt['train']['n_iter']: break

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            
            if current_step % 100 == 0:
                logs = diffusion.get_current_log()
                wandb.log({'train/loss': logs['loss_total']}, step=current_step)

            if current_step % opt['train']['val_freq'] == 0:
                diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                avg_rmsd = 0.0
                avg_rmsd_base = 0.0
                count = 0
                
                # Loop over the SMALL subset of validation data
                for val_data in val_loader:
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    
                    pred_residual = _final_frame(visuals['SR'])
                    baseline_norm = visuals['LR_UP']
                    sr_reconstructed = baseline_norm + pred_residual
                    
                    hr_norm = visuals['HR']
                    ct_gpu = visuals['CT']
                    
                    sr_clamped = sr_reconstructed.clamp(-1.0, 1.0)
                    hr_clamped = hr_norm.clamp(-1.0, 1.0)
                    base_clamped = baseline_norm.clamp(-1.0, 1.0)
                    mask = ct_body_mask(ct_gpu.cpu())
                    
                    avg_rmsd += _rmsd(sr_clamped.cpu(), hr_clamped.cpu(), mask)
                    avg_rmsd_base += _rmsd(base_clamped.cpu(), hr_clamped.cpu(), mask)
                    count += 1
                
                if count > 0:
                    wandb.log({
                        'val/rmsd_mask': avg_rmsd / count, 
                        'val/baseline': avg_rmsd_base / count
                    }, step=current_step)
                
                diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

if __name__ == '__main__':
    train()