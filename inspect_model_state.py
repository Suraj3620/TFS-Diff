import argparse
import torch
import numpy as np
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

    # 1. FORCE THE EXACT PATH TO YOUR CHECKPOINT
    ckpt_path = "/home/sdesai/THESIS/TFS-Diff-main/experiments/CT_Dose_3D_UNet_Baseline_260222_020302/experiments/CT_Dose_3D_UNet_Baseline/checkpoints/I2000_E1_gen.pth"
    
    print(f"[*] Loading Model Checkpoint: {ckpt_path}")

    # 2. Load Model
    diffusion = Model.create_model(opt)
    
    # Force load the state dict directly to guarantee it loads
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if isinstance(diffusion.netG, torch.nn.DataParallel):
        diffusion.netG.module.load_state_dict(state_dict, strict=False)
    else:
        diffusion.netG.load_state_dict(state_dict, strict=False)
        
    diffusion.netG.eval()

    # 3. Load 1 Batch of Validation Data
    print("[*] Loading 1 Batch of Validation Data...")
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    
    batch = next(iter(val_loader))
    
    # Extract tensors
    cond = batch['SR'].to(diffusion.device) # [Baseline, CT]
    hr = batch['HR'].to(diffusion.device)   # Ground Truth
    
    print("="*50)
    print("      RAW NETWORK OUTPUT ANALYSIS")
    print("="*50)

    with torch.no_grad():
        # We test 3 different timesteps (t=999, t=500, t=0) to see what the network predicts
        timesteps_to_test = [999, 500, 0]
        
        for t_val in timesteps_to_test:
            # Create timestep tensor
            t = torch.full((cond.shape[0],), t_val, device=diffusion.device, dtype=torch.long)
            
            # Simulate the noisy input (x_t) at this timestep
            noise = torch.randn_like(hr)
            x_noisy = diffusion.netG.module.q_sample(x_start=hr, t=t, noise=noise) if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG.q_sample(x_start=hr, t=t, noise=noise)
            
            # Feed to network: [Noisy_GT, Baseline, CT]
            model_input = torch.cat([x_noisy, cond], dim=1)
            
            # GET RAW PREDICTION (The "Correction")
            network_module = diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG
            raw_prediction = network_module.denoise_fn(model_input, t)
            
            # Stats
            raw_np = raw_prediction.cpu().numpy()
            
            print(f"--- Timestep {t_val} ---")
            print(f"  Min Value:  {raw_np.min():.6f}")
            print(f"  Max Value:  {raw_np.max():.6f}")
            print(f"  Mean (Abs): {np.mean(np.abs(raw_np)):.6f}")
            
            # Ideal Target Analysis
            # The ideal prediction should be exactly (GT - Baseline)
            baseline = cond[:, 0:1, ...]
            ideal_target = (hr - baseline).cpu().numpy()
            
            print(f"  [IDEAL] Min: {ideal_target.min():.6f}, Max: {ideal_target.max():.6f}, Mean(Abs): {np.mean(np.abs(ideal_target)):.6f}")
            
            # If the network output is wildly out of scale compared to the ideal target, training has exploded.
            if np.max(np.abs(raw_np)) > 2.0:
                print("  => WARNING: Network output is MASSIVE. Gradients likely exploded.")
            print("")

if __name__ == "__main__":
    main()