import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import data as Data
import model as Model
import core.logger as Logger
import os

def extract_middle_slice(tensor):
    """Safely extracts the middle 2D slice from a 3D/4D/5D tensor for visualization."""
    t = tensor.detach().cpu().squeeze().numpy()
    if t.ndim == 3: return t[t.shape[0]//2, :, :]
    elif t.ndim == 2: return t
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_3d_unet_train.json')
    parser.add_argument('-p', '--phase', type=str, default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    print("="*50)
    print("1. DATASET JUNCTION (Checking what the loader produces)")
    print("="*50)
    
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    batch = next(iter(val_loader))
    
    # Extract tensors based on our new Pure SR3 setup
    target_hr = batch['RES']  # Should now be the FULL DOSE
    cond = batch['SR']        # [Baseline, CT]
    baseline = cond[:, 0:1]
    ct_scan = cond[:, 1:2]
    
    print(f"Target (RES) Shape: {target_hr.shape} | Range: [{target_hr.min():.4f}, {target_hr.max():.4f}]")
    print(f"Baseline     Shape: {baseline.shape} | Range: [{baseline.min():.4f}, {baseline.max():.4f}]")
    print(f"CT Scan      Shape: {ct_scan.shape} | Range: [{ct_scan.min():.4f}, {ct_scan.max():.4f}]")
    
    if torch.allclose(target_hr, baseline, atol=1e-4):
        print("[!] ERROR: Target is identical to Baseline! Identity trap is still active.")
    else:
        print("[+] PASS: Target and Baseline are different.")

    print("\n" + "="*50)
    print("2. FORWARD TRAINING JUNCTION (Checking what the network sees)")
    print("="*50)
    
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    device = diffusion.device
    
    # Simulate a training step at t=500 (Halfway corrupted)
    t_500 = torch.tensor([500], device=device, dtype=torch.long)
    target_hr_device = target_hr.to(device)
    noise = torch.randn_like(target_hr_device)
    
    # Add noise to the Target
    x_noisy = diffusion.netG.module.q_sample(x_start=target_hr_device, t=t_500, noise=noise) if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG.q_sample(x_start=target_hr_device, t=t_500, noise=noise)
    
    # Network Input
    cond_device = cond.to(device)
    network_input = torch.cat([cond_device, x_noisy], dim=1)
    
    print(f"Network Input Shape: {network_input.shape} (Should be 3 Channels: Base, CT, Noisy_Target)")
    print(f"Noisy Target Range: [{x_noisy.min():.4f}, {x_noisy.max():.4f}]")

    print("\n" + "="*50)
    print("3. REVERSE INFERENCE JUNCTION (Checking the generation loop)")
    print("="*50)
    
    diffusion.netG.eval()
    
    # We will manually run the reverse loop to save intermediate images
    ddpm = diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG
    
    b = cond_device.shape[0]
    spatial_dims = target_hr.shape[2:]
    
    # Start from PURE NOISE
    img_t = torch.randn((b, 1, *spatial_dims), device=device)
    
    saved_states = {}
    saved_states['t=1000 (Start)'] = img_t.clone()
    
    print("[*] Running Reverse Diffusion...")
    with torch.no_grad():
        for i in reversed(range(0, ddpm.num_timesteps)):
            t = torch.tensor([i], device=device, dtype=torch.long)
            img_t = ddpm.p_sample(img_t, t, cond=cond_device, clip_denoised=True)
            
            if i == 750: saved_states['t=750'] = img_t.clone()
            if i == 500: saved_states['t=500'] = img_t.clone()
            if i == 250: saved_states['t=250'] = img_t.clone()
            if i == 0:   saved_states['t=0 (Final)'] = img_t.clone()

    print("[+] Reverse loop complete.")

    print("\n" + "="*50)
    print("4. GENERATING VISUAL PROOF")
    print("="*50)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Plot Inputs
    axes[0].imshow(extract_middle_slice(ct_scan), cmap='gray')
    axes[0].set_title("Input Channel 1: CT Scan")
    
    axes[1].imshow(extract_middle_slice(baseline), cmap='gray')
    axes[1].set_title("Input Channel 2: Baseline (Guide)")
    
    axes[2].imshow(extract_middle_slice(target_hr), cmap='gray')
    axes[2].set_title("Target: Full GT Dose")
    
    axes[3].imshow(extract_middle_slice(x_noisy), cmap='gray')
    axes[3].set_title("Training: Network sees this at t=500")

    # Plot Reverse Process
    axes[4].imshow(extract_middle_slice(saved_states['t=1000 (Start)']), cmap='gray')
    axes[4].set_title("Inference Start: Pure Noise (t=1000)")
    
    axes[5].imshow(extract_middle_slice(saved_states['t=500']), cmap='gray')
    axes[5].set_title("Inference Step: t=500")
    
    axes[6].imshow(extract_middle_slice(saved_states['t=250']), cmap='gray')
    axes[6].set_title("Inference Step: t=250")
    
    axes[7].imshow(extract_middle_slice(saved_states['t=0 (Final)']), cmap='gray')
    axes[7].set_title("Final Model Output (t=0)")

    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('data_flow_proof.png', dpi=150)
    print("[*] Saved 'data_flow_proof.png'. OPEN THIS FILE.")

if __name__ == "__main__":
    main()