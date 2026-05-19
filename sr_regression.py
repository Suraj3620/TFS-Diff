import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import data as Data
from model.unet_3d import UNet3D
from core.metrics_bodymask import ct_body_mask
import core.logger as Logger

# --- Helper: Save Visuals with Heatmap ---
def _save_montage(save_dir, tag, ct, base, pred_res, hr):
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract middle slice
    def to_slice(x):
        x = x.detach().cpu().numpy().squeeze()
        if x.ndim == 3: return x[x.shape[0]//2]
        return x

    s_ct = to_slice(ct)
    s_base = to_slice(base)
    s_res = to_slice(pred_res)
    s_hr = to_slice(hr)
    
    # Reconstruct SR for visualization
    s_sr = s_base + s_res

    # Normalize for PNG [0, 255]
    def norm(x):
        mi, ma = x.min(), x.max()
        if ma - mi < 1e-8: return np.zeros_like(x, dtype=np.uint8)
        return ((x - mi) / (ma - mi) * 255).astype(np.uint8)

    img_ct = cv2.cvtColor(norm(s_ct), cv2.COLOR_GRAY2BGR)
    img_base = cv2.cvtColor(norm(s_base), cv2.COLOR_GRAY2BGR)
    img_sr = cv2.cvtColor(norm(s_sr), cv2.COLOR_GRAY2BGR)
    img_hr = cv2.cvtColor(norm(s_hr), cv2.COLOR_GRAY2BGR)
    
    # Residual Heatmap (Prediction)
    # Map [-1, 1] to Color
    res_norm = np.clip((s_res + 0.5), 0, 1) # Center 0.0 at 0.5 gray
    img_res_heat = cv2.applyColorMap((res_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Strip: [CT | Base | SR | HR | Predicted_Residual]
    strip = cv2.hconcat([img_ct, img_base, img_sr, img_hr, img_res_heat])
    cv2.imwrite(os.path.join(save_dir, f"{tag}.png"), strip)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_3d_unet_train.json')
    parser.add_argument('-p', '--phase', type=str, default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    
    # Logger args
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('--log_eval', action='store_true')
    parser.add_argument('--log_infer', action='store_true')
    
    args = parser.parse_args()
    
    # 1. Load Config
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # Paths
    exp_root = opt['path']['experiments_root'].replace("_UNet_Baseline", "_UNet_REGRESSION_RESIDUAL")
    os.makedirs(exp_root, exist_ok=True)
    log_dir = os.path.join(exp_root, 'logs')
    res_dir = os.path.join(exp_root, 'results')
    ckpt_dir = os.path.join(exp_root, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    Logger.setup_logger(None, log_dir, 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    writer = SummaryWriter(log_dir=os.path.join(exp_root, 'tb_logger'))
    
    if opt['enable_wandb']:
        import wandb
        wandb.init(project="3D_Dose_Regression", config=opt, name="Regression_Clean_Residual")

    # 2. Data Loaders
    opt['datasets']['train']['batch_size'] = 4
    opt['datasets']['train']['num_workers'] = 4
    opt['datasets']['val']['batch_size'] = 4
    
    train_set = Data.create_dataset(opt['datasets']['train'], 'train')
    train_loader = Data.create_dataloader(train_set, opt['datasets']['train'], 'train')
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    
    logger.info(f"Dataset Size: {len(train_set)} patches")

    # 3. Model Setup (Standard U-Net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D(
        in_channel=2, 
        out_channel=1, 
        inner_channel=64,
        res_blocks=2,
        image_size=32
    ).to(device)
    
    # ZERO INITIALIZATION (Crucial for Identity Start)
    if hasattr(model, 'conv_out'):
        nn.init.zeros_(model.conv_out[-1].weight)
        nn.init.zeros_(model.conv_out[-1].bias)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()
    
    logger.info("Starting Regression on UNSCALED Residuals...")
    
    step = 0
    best_rmsd = 1.0
    val_freq = 1000 

    # 4. Training Loop
    for epoch in range(100):
        model.train()
        for i, batch in enumerate(train_loader):
            step += 1
            
            # Input: [Baseline, CT]
            inputs = batch['SR'].to(device)
            # Target: Residual (GT - Baseline)
            target_res = batch['RES'].to(device)
            
            # Dummy Time (0)
            t = torch.zeros(inputs.size(0), device=device)
            
            # Forward
            pred_res = model(inputs, t)
            
            # Loss
            loss = criterion(pred_res, target_res)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                logger.info(f"Step {step}: Loss: {loss.item():.6f}")
                writer.add_scalar('train/loss', loss.item(), step)
                if opt['enable_wandb']: wandb.log({'train/loss': loss.item()}, step=step)
                
            # 5. Validation
            if step % val_freq == 0:
                model.eval()
                val_rmsd_list = []
                val_base_list = []
                
                with torch.no_grad():
                    for j, vbatch in enumerate(val_loader):
                        if j >= 10: break 
                        
                        v_inputs = vbatch['SR'].to(device) # [Base, CT]
                        v_hr = vbatch['HR'].to(device)     # Full GT Dose
                        v_base = vbatch['LR'].to(device)   # Full Baseline
                        
                        v_t = torch.zeros(v_inputs.size(0), device=device)
                        
                        # Predict Residual
                        v_pred_res = model(v_inputs, v_t)
                        
                        # Reconstruct: SR = Baseline + Predicted_Residual
                        # NO SCALING FACTOR HERE
                        v_sr = v_base + v_pred_res
                        
                        # Clamp [-1, 1]
                        v_sr = v_sr.clamp(-1.0, 1.0)
                        
                        # Metrics
                        mask = ct_body_mask(v_inputs[:, 1:2]) 
                        
                        diff2 = (v_sr - v_hr) ** 2
                        rmsd = torch.sqrt((diff2 * mask).sum() / mask.sum().clamp_min(1))
                        val_rmsd_list.append(rmsd.item())
                        
                        diff2_base = (v_base - v_hr) ** 2
                        rmsd_base = torch.sqrt((diff2_base * mask).sum() / mask.sum().clamp_min(1))
                        val_base_list.append(rmsd_base.item())
                        
                        if j == 0:
                            _save_montage(res_dir, f"step_{step}", 
                                          v_inputs[0, 1], v_base[0], 
                                          v_pred_res[0], v_hr[0])

                avg_rmsd = np.mean(val_rmsd_list)
                avg_base = np.mean(val_base_list)
                
                logger.info(f"VAL | RMSD: {avg_rmsd:.6f} | Base: {avg_base:.6f}")
                
                if opt['enable_wandb']:
                    wandb.log({'val/rmsd': avg_rmsd, 'val/baseline': avg_base}, step=step)

                if avg_rmsd < best_rmsd:
                    best_rmsd = avg_rmsd
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                    logger.info("New Best RMSD.")
                
                model.train()

if __name__ == "__main__":
    main()