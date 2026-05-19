import os
import argparse
import logging
import torch
import cv2
import numpy as np

import data as Data
import model as Model
import core.logger as Logger

# --- Helper functions ---
def _robust_lohi(x, p1=1, p99=99):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().float().numpy()
    x = np.asarray(x).squeeze()
    lo, hi = np.percentile(x, [p1, p99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    return lo, hi

def _to_uint8_gray(x, lohi=None):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().float().numpy()
    x = np.asarray(x).squeeze()
    if lohi is None: lo, hi = _robust_lohi(x)
    else: lo, hi = lohi
    y = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    y = (y * 255.0).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)

def _annotate(img_rgb, text):
    out = img_rgb.copy()
    cv2.putText(out, str(text), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, str(text), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

def _final_frame(t):
    if isinstance(t, (list, tuple)): return t[-1]
    if torch.is_tensor(t) and t.dim() == 5: return t[-1]
    return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_train.json', help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None, help="Comma-separated GPU ids")
    parser.add_argument('--output_dir', type=str, default='experiments/inference_results', help="Directory to save result images")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of validation samples to infer and save")

    args = parser.parse_args()
    args.phase = 'val'
    args.enable_wandb = False
    args.debug = False
    args.log_wandb_ckpt = False
    args.log_eval = False
    args.log_infer = False

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    Logger.setup_logger(None, opt['path']['log'], 'inference', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info("Starting Inference...")

    diffusion = Model.create_model(opt)
    logger.info('Model created.')

    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    logger.info(f'Validation dataset loaded with {len(val_set)} samples.')

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    diffusion.netG.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    
    count = 0
    for val_data in val_loader:
        if count >= args.num_samples:
            break
        
        logger.info(f"Inferring sample {count + 1}/{args.num_samples}...")
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()

        sr_final = _final_frame(visuals['SR'])
        
        # Get single-item tensors for visualization
        ct_img = visuals['CT'][0]
        lr_up_img = visuals['LR_UP'][0]
        sr_img = sr_final[0]
        hr_img = visuals['HR'][0]
        
        # Get a consistent window for dose images based on ground truth
        dose_lohi = _robust_lohi(hr_img)
        
        # Convert all to annotated RGB images
        ct_viz = _annotate(_to_uint8_gray(ct_img), "CT")
        lr_up_viz = _annotate(_to_uint8_gray(lr_up_img, lohi=dose_lohi), "LR-up (Baseline)")
        sr_viz = _annotate(_to_uint8_gray(sr_img, lohi=dose_lohi), "SR (Model Output)")
        hr_viz = _annotate(_to_uint8_gray(hr_img, lohi=dose_lohi), "HR (Ground Truth)")
        
        montage = cv2.hconcat([ct_viz, lr_up_viz, sr_viz, hr_viz])
        
        uid = val_data.get('uid', ['unknown'])[0]
        slice_num = val_data.get('slice', [-1])[0].item()
        save_path = os.path.join(args.output_dir, f'inference_uid_{uid}_slice_{slice_num}.png')
        cv2.imwrite(save_path, montage)
        logger.info(f"Saved montage to {save_path}")

        count += 1

    logger.info("Inference complete.")