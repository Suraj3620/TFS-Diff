import argparse
import torch
import torch.nn as nn
import numpy as np
import data as Data
import model as Model
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

# --- PROPER PYTORCH MODULE FOR THE ZERO-TEST ---
class FakeDenoise(nn.Module):
    def forward(self, x, t):
        # Return a tensor of zeros matching the target output shape (1 channel)
        return torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_3d_dit_train.json')
    parser.add_argument('-p', '--phase', type=str, default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('--log_eval', action='store_true')
    parser.add_argument('--log_infer', action='store_true')
    
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    print("[*] Loading Validation Set...")
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')

    print("[*] Creating Diffusion Model...")
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    # ---------------------------------------------------------
    # THE LOBOTOMY: Force the network to ALWAYS predict 0.0
    # ---------------------------------------------------------
    fake_denoise = FakeDenoise().to(diffusion.device)

    # Override the actual neural network with our fake module
    if isinstance(diffusion.netG, torch.nn.DataParallel):
        diffusion.netG.module.denoise_fn = fake_denoise
    else:
        diffusion.netG.denoise_fn = fake_denoise
    # ---------------------------------------------------------

    print("[*] Running 1000-Step Diffusion Loop with Zeroed Network...")
    
    rmsd_list = []
    
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            if i >= 3: break # Test 3 samples
            
            # Feed data and run standard test loop
            diffusion.feed_data(val_data)
            diffusion.test(continous=False) 
            
            visuals = diffusion.get_current_visuals()
            
            # Extract final SR image and GT
            sr_final = visuals['SR'][-1] if isinstance(visuals['SR'], list) else visuals['SR']
            hr_gt = visuals['HR']
            baseline = visuals['LR_UP']
            ct = visuals['CT']
            
            mask = ct_body_mask(ct.cpu())
            
            # Clamp logic (matches validation)
            sr_clamped = sr_final.clamp(-1.0, 1.0).cpu()
            hr_clamped = hr_gt.clamp(-1.0, 1.0).cpu()
            
            val_rmsd = _rmsd(sr_clamped, hr_clamped, mask)
            base_rmsd = _rmsd(baseline.clamp(-1.0, 1.0).cpu(), hr_clamped, mask)
            
            rmsd_list.append(val_rmsd)
            print(f"Sample {i}:")
            print(f"  -> DDPM Output RMSD: {val_rmsd:.6f}")
            print(f"  -> True Baseline RMSD: {base_rmsd:.6f}")

    print("="*40)
    print(f"AVERAGE DDPM ZERO-WEIGHT RMSD: {np.mean(rmsd_list):.6f}")
    print("="*40)

if __name__ == "__main__":
    main()