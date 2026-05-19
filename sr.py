import os
import argparse
import logging
import numpy as np
import cv2
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import data as Data
import model as Model
import core.logger as Logger
from core.metrics_bodymask import ct_body_mask

try:
    from tensorboardX import SummaryWriter
except Exception:
    from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.benchmark = True
try: set_start_method('spawn')
except RuntimeError: pass
mp.set_sharing_strategy("file_system")

def _final_frame(t):
    if isinstance(t, (list, tuple)): return t[-1]
    if torch.is_tensor(t) and t.dim() == 6: return t[-1]
    return t

def _rmsd(a, b, mask=None):
    a, b = a.float(), b.float()
    diff2 = (a - b) ** 2
    if mask is not None: mse = (diff2 * mask).sum() / mask.sum().clamp_min(1.0)
    else: mse = diff2.mean()
    return torch.sqrt(mse.clamp_min(1e-12)).item()

def _psnr(a, b, mask=None):
    a, b = a.float(), b.float()
    diff2 = (a - b) ** 2
    if mask is not None: mse = (diff2 * mask).sum() / mask.sum().clamp_min(1.0)
    else: mse = diff2.mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-12)).item()

def _save_per_sample_visuals(save_dir, tag, ct, lr, sr, hr):
    os.makedirs(save_dir, exist_ok=True)
    def to_2d(t): return t[:, :, t.shape[2]//2, :, :].detach().cpu().squeeze().numpy()

    ct_2d = to_2d(ct)
    base_2d = to_2d(lr)
    sr_2d = to_2d(sr)
    hr_2d = to_2d(hr)

    # Lock scales to GT
    hr_min, hr_max = hr_2d.min(), hr_2d.max()
    if hr_max - hr_min < 1e-9: hr_max = hr_min + 1e-9

    def norm_to_hr(img):
        return (np.clip((img - hr_min) / (hr_max - hr_min), 0, 1) * 255.0).astype(np.uint8)

    ct_specific = ((ct_2d - ct_2d.min()) / (ct_2d.max() - ct_2d.min() + 1e-8) * 255).astype(np.uint8)
    base_specific = norm_to_hr(base_2d)
    sr_specific = norm_to_hr(sr_2d)
    hr_specific = norm_to_hr(hr_2d)

    diff = np.abs(hr_2d - sr_2d)
    diff_norm = np.clip(diff / (hr_max * 0.10 + 1e-8), 0, 1)
    heatmap_bgr = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    strip = cv2.hconcat([cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in [ct_specific, base_specific, sr_specific, hr_specific]] + [heatmap_bgr])
    cv2.imwrite(os.path.join(save_dir, f'montage_{tag}.png'), strip)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_train.json')
    parser.add_argument('-p', '--phase', type=str, default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--vis_per_batch', type=int, default=8)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('--log_eval', action='store_true')
    parser.add_argument('--log_infer', action='store_true')
    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    if opt['enable_wandb']:
        import wandb
        wandb.init(project="3D_Dose_Diffusion", config=opt, name="Diffusion_Noise_Pred")

    train_set = Data.create_dataset(opt['datasets']['train'], 'train')
    train_loader = Data.create_dataloader(train_set, opt['datasets']['train'], 'train')
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    
    diffusion = Model.create_model(opt)

    if opt['phase'] == 'train':
        current_step, current_epoch = diffusion.begin_step, diffusion.begin_epoch
        best_val_rmsd = float('inf')
        
        while current_step < opt['train']['n_iter']:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters(current_step)

                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    msg = f"<epoch:{current_epoch:3d}, iter:{current_step:8,d}> loss: {logs['loss_total']:.4e}"
                    tb_logger.add_scalar('train/loss', logs['loss_total'], current_step)
                    logger.info(msg)
                    if opt['enable_wandb']: wandb.log({'train/loss': logs['loss_total']}, step=current_step)

                if current_step % opt['train']['val_freq'] == 0:
                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    save_root = os.path.join(opt['path']['results'], f"val_E{current_epoch:03d}_I{current_step:07d}")

                    # ADDED 'val_loss' TO TRACKING
                    val_metrics = {'rmsd':[], 'base_rmsd': [], 'val_loss':[]}
                    
                    for b_idx, val_data in enumerate(val_loader):
                        if b_idx >= 5: break
                        
                        diffusion.feed_data(val_data)
                        
                        # --- THE FIX: Compute Validation Loss ---
                        with torch.no_grad():
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                v_loss = diffusion.netG(diffusion.data)
                                val_metrics['val_loss'].append(v_loss.item())
                        # ----------------------------------------

                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()

                        pred_residual = _final_frame(visuals['SR'])
                        baseline = visuals['LR_UP']
                        ct_gpu = visuals['CT']
                        hr_norm = visuals['HR']
                        
                        # Reconstruct SR = Baseline + Residual
                        sr_reconstructed = baseline + pred_residual

                        sr_c = sr_reconstructed.clamp(-1.0, 1.0)
                        hr_c = hr_norm.clamp(-1.0, 1.0)
                        base_c = baseline.clamp(-1.0, 1.0)
                        mask = ct_body_mask(ct_gpu.cpu())

                        rmsd = _rmsd(sr_c.cpu(), hr_c.cpu(), mask)
                        base_rmsd = _rmsd(base_c.cpu(), hr_c.cpu(), mask)

                        val_metrics['rmsd'].append(rmsd)
                        val_metrics['base_rmsd'].append(base_rmsd)

                        if b_idx == 0:
                            _save_per_sample_visuals(save_root, "val_0", ct_gpu[0:1], baseline[0:1], sr_reconstructed[0:1], hr_c[0:1])

                    # --- THE FIX: Log Val Loss to Tensorboard ---
                    avg_val_loss = np.mean(val_metrics['val_loss'])
                    avg_rmsd = np.mean(val_metrics['rmsd'])
                    avg_base = np.mean(val_metrics['base_rmsd'])

                    logger.info(f"VAL | Loss: {avg_val_loss:.6f} | RMSD: {avg_rmsd:.6f}, Base: {avg_base:.6f}")
                    tb_logger.add_scalar('val/loss', avg_val_loss, current_step)
                    tb_logger.add_scalar('val/rmsd', avg_rmsd, current_step)
                    
                    if opt['enable_wandb']: 
                        wandb.log({'val/loss': avg_val_loss, 'val/rmsd': avg_rmsd, 'val/baseline': avg_base}, step=current_step)

                    if avg_rmsd < best_val_rmsd:
                        best_val_rmsd = avg_rmsd
                        diffusion.save_network(current_epoch, current_step)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')