import os
import argparse
import logging
import torch
import cv2
import numpy as np

import data as Data
import model as Model
import core.logger as Logger

# --- Helper functions for creating nice visualizations ---
# (These are copied directly from the main sr.py script)

def _robust_lohi(x, p1=1, p99=99):
    """Get robust (lo,hi) percentiles from a tensor/array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        if x.dim() == 4: x = x[0]
        if x.dim() == 3: x = x[0]
        x = x.numpy()
    x = np.asarray(x)
    lo, hi = np.percentile(x, [p1, p99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            lo, hi = 0.0, 1.0
    return lo, hi

def _to_uint8_gray(x, p1=1, p99=99, lohi=None, invert=False):
    """Map to uint8 using either provided (lo,hi) or robust percentiles; return RGB."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        if x.dim() == 4: x = x[0]
        if x.dim() == 3: x = x[0]
        x = x.numpy()
    x = np.asarray(x)
    if lohi is None:
        lo, hi = _robust_lohi(x, p1, p99)
    else:
        lo, hi = lohi
    y = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    if invert:
        y = 1.0 - y
    y = (y * 255.0).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)

def _annotate(img_rgb, text, pos=(6, 16)):
    """Put a small label on an RGB image."""
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    out = img_rgb.copy()
    cv2.putText(out, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),   1, cv2.LINE_AA)
    return out
    
def _final_frame(t):
    if isinstance(t, (list, tuple)): return t[-1]
    if torch.is_tensor(t) and t.dim() == 5: return t[-1]
    return t

def _save_per_sample_visuals(save_dir, tag, ct, lr, sr, hr, base01=None, invert_sr=False):
    os.makedirs(save_dir, exist_ok=True)
    dose_lohi = _robust_lohi(hr)
    ct_img = _to_uint8_gray(ct)
    lr_img = _to_uint8_gray(lr, lohi=dose_lohi, invert=False)
    sr_img = _to_uint8_gray(sr, lohi=dose_lohi, invert=invert_sr)
    hr_img = _to_uint8_gray(hr, lohi=dose_lohi, invert=False)

    ct_img = _annotate(ct_img, "CT (Zeroed Out)")
    lr_img = _annotate(lr_img, "LR-up (Input)")
    sr_img = _annotate(sr_img, "SR (No-CT Prediction)")
    hr_img = _annotate(hr_img, "HR (Ground Truth)")

    mon = cv2.hconcat([ct_img, lr_img, sr_img, hr_img])
    cv2.imwrite(os.path.join(save_dir, f'montage_ablation_{tag}.png'), mon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_train.json', help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None, help="Comma-separated GPU ids")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of validation samples to test")
    parser.add_argument('--output_dir', type=str, default='experiments/ablation_study_results', help="Directory to save result images")

    args = parser.parse_args()
    args.phase = 'val'
    args.enable_wandb = False
    args.debug = False
    args.log_wandb_ckpt = False
    args.log_eval = False
    args.log_infer = False
    
    # Load the same configuration used for training to ensure the model architecture matches.
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    Logger.setup_logger(None, opt['path']['log'], 'ablation_test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info("Starting CT Ablation Study...")

    # Create the pre-trained model.
    diffusion = Model.create_model(opt)
    logger.info('Model created successfully.')

    # Load the validation dataset. We'll run the test on these samples.
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    logger.info(f'Validation dataset loaded with {len(val_set)} samples.')

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

    diffusion.netG.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    
    count = 0
    for b_idx, val_data in enumerate(val_loader):
        if count >= args.num_samples:
            break
            
        # This is the core of the experiment. We take the 2-channel conditioning input,
        # which is (LR_dose, CT_scan), and we overwrite the CT channel with a constant value (-1).
        # This effectively removes the CT information, forcing the model to predict without it.
        logger.info(f"Processing batch {b_idx+1}...")
        val_data['SR'][:, 1, :, :] = -1 # Zero out the CT channel.

        # Now, we run inference as usual, but with the modified input.
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()

        sr_final = _final_frame(visuals['SR'])
        
        # Save a visual comparison for each sample in the batch.
        for i in range(sr_final.shape[0]):
            if count >= args.num_samples:
                break

            uid = val_data['uid'][i] if 'uid' in val_data else f"b{b_idx}_i{i}"
            sl = val_data['slice'][i] if 'slice' in val_data else "N/A"
            tag = f"uid{uid}_slice{sl}"
            
            _save_per_sample_visuals(
                save_dir=args.output_dir,
                tag=tag,
                ct=visuals['CT'][i:i+1],
                lr=visuals['LR_UP'][i:i+1],
                sr=sr_final[i:i+1],
                hr=visuals['HR'][i:i+1]
            )
            count += 1
            logger.info(f"Saved visualization for sample {count}/{args.num_samples}: {tag}")

    logger.info(f"Ablation study complete. Results are saved in: {args.output_dir}")