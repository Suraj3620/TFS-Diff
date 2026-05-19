import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import data as Data
import model as Model
import core.logger as Logger

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

    # 1. UPDATE THIS PATH TO YOUR LATEST TRAINED ATTENTION MODEL CHECKPOINT
    ckpt_path = '/home/sdesai/THESIS/TFS-Diff-main/experiments/CT_Dose_3D_UNet_Fast_L2_Residual_260315_205122/experiments/CT_Dose_3D_UNet_Fast/checkpoints/I6000_E2_gen.pth' # <-- UPDATE ME!
    opt['path']['resume_state'] = ckpt_path
    
    print(f"[*] Loading Model Checkpoint: {ckpt_path}")

    # 2. Load Model & Data
    diffusion = Model.create_model(opt)
    diffusion.netG.eval()
    
    print("[*] Loading Validation Set...")
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    
    batch = next(iter(val_loader))
    
    print("[*] Running Diffusion Inference...")
    with torch.no_grad():
        diffusion.feed_data(batch)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()
        
        # --- EXTRACT ATTENTION WEIGHTS ---
        # Navigate through the DataParallel/GaussianDiffusion wrappers to the UNet bottleneck
        unet = diffusion.netG.module.denoise_fn if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG.denoise_fn
        
        if hasattr(unet, 'cross_attn') and unet.cross_attn.attn_weights is not None:
            # attn_weights shape:[B, num_heads, N_queries, N_keys]
            attn_raw = unet.cross_attn.attn_weights
            
            # Sum over queries (how much attention did each CT voxel receive total?) and average heads
            # Shape becomes:[B, N_keys]
            attn_keys = attn_raw.sum(dim=2).mean(dim=1)
            
            # Reshape back to 3D spatial bottleneck size. 
            # 32x32x32 image with 3 downsamples (mults 1,2,4,8) = 4x4x4 bottleneck
            bottleneck_dim = int(round(attn_keys.shape[-1] ** (1/3)))
            attn_spatial = attn_keys.view(1, 1, bottleneck_dim, bottleneck_dim, bottleneck_dim)
            
            # Upsample back to 32x32x32 to match the CT scan
            attn_up = F.interpolate(attn_spatial, size=(32, 32, 32), mode='trilinear', align_corners=False)
            attn_map_3d = attn_up[0, 0].cpu().numpy()
        else:
            print("[!] Warning: Cross Attention weights not found. Did you run the Attn model?")
            attn_map_3d = np.zeros((32, 32, 32))
            
    # Extract middle slice from the 3D volume
    def get_slice(tensor):
        t = tensor.detach().cpu().squeeze().numpy()
        if t.ndim == 3: return t[t.shape[0]//2]
        return t

    hr = get_slice(visuals['HR'])
    base = get_slice(visuals['LR_UP'])
    sr_res = get_slice(visuals['SR']) # Diffusion output is the predicted residual
    ct = get_slice(visuals['CT'])
    
    sr = base + sr_res # Full reconstructed image
    attn_slice = attn_map_3d[attn_map_3d.shape[0]//2]

    # --- CALCULATION OF TRUE VS PREDICTED RESIDUALS ---
    true_residual = hr - base
    predicted_residual = sr_res
    
    # Normalize Attention Map for display [0, 1]
    attn_slice = (attn_slice - attn_slice.min()) / (attn_slice.max() - attn_slice.min() + 1e-8)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 14))
    vmax_res = max(np.abs(true_residual).max(), np.abs(predicted_residual).max())

    # Row 1: The Core Images & Attention
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(ct, cmap='bone')
    ax1.set_title("CT Scan (Anatomy)")
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(ct, cmap='bone')
    ax2.imshow(attn_slice, cmap='jet', alpha=0.4) # Overlay Attention
    ax2.set_title("CT Gating\n(Model Attention Overlay)")

    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(sr, cmap='gray', vmin=0, vmax=hr.max())
    ax3.set_title("Super-Res (Model Dose)")

    ax4 = plt.subplot(3, 4, 4)
    ax4.imshow(hr, cmap='gray', vmin=0, vmax=hr.max())
    ax4.set_title("Ground Truth (Target Dose)")

    # Row 2: The Deltas (Physics Corrections)
    ax5 = plt.subplot(3, 3, 4)
    im5 = ax5.imshow(true_residual, cmap='bwr', vmin=-vmax_res, vmax=vmax_res)
    ax5.set_title("Target Residual\n(What it SHOULD learn)")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = plt.subplot(3, 3, 5)
    im6 = ax6.imshow(predicted_residual, cmap='bwr', vmin=-vmax_res, vmax=vmax_res)
    ax6.set_title("Predicted Residual\n(What it ACTUALLY learned)")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = plt.subplot(3, 3, 6)
    final_error = hr - sr
    im7 = ax7.imshow(final_error, cmap='bwr', vmin=-vmax_res, vmax=vmax_res)
    ax7.set_title("Final Remaining Error\n(HR - SR)")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # Row 3: Enhanced 1D Physics Profile (The Professor's Request)
    mid_y = hr.shape[0] // 2
    
    ax8 = plt.subplot(3, 1, 3)
    
    # Left Y-Axis: Dose
    color_gt, color_base, color_sr = 'black', 'blue', 'red'
    ax8.plot(hr[mid_y, :], label='Ground Truth Dose', color=color_gt, linewidth=2, linestyle='--')
    ax8.plot(base[mid_y, :], label='Baseline Dose (Interpolated)', color=color_base, alpha=0.6)
    ax8.plot(sr[mid_y, :], label='TFSDiff Model Dose', color=color_sr, linewidth=2, alpha=0.9)
    ax8.set_ylabel("Normalized Dose Level", color=color_gt, fontsize=12)
    ax8.tick_params(axis='y', labelcolor=color_gt)
    ax8.set_xlabel("X-Axis Pixel Coordinates", fontsize=12)
    ax8.legend(loc='upper left')

    # Right Y-Axis: CT Anatomy (Bone Density)
    ax9 = ax8.twinx()
    color_ct = 'green'
    # Smooth CT slightly for a cleaner 1D profile
    ct_line = np.convolve(ct[mid_y, :], np.ones(3)/3, mode='same')
    ax9.plot(ct_line, label='CT Anatomy (Density)', color=color_ct, linewidth=1.5, alpha=0.5)
    ax9.fill_between(range(len(ct_line)), ct_line, ct_line.min(), color=color_ct, alpha=0.1)
    ax9.set_ylabel("CT Density (HU Normalized)", color=color_ct, fontsize=12)
    ax9.tick_params(axis='y', labelcolor=color_ct)
    ax9.legend(loc='upper right')

    plt.title(f"1D Multi-Modal Physics Profile (Y={mid_y})\nObserve how dose drops exactly where CT density (bone) peaks.", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('clinical_proof_analysis_attention.png', dpi=200)
    print("[*] Saved 'clinical_proof_analysis_attention.png'. Open this to see the undeniable proof!")

if __name__ == "__main__":
    main()