import os
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import data as Data
import model as Model
import core.logger as Logger
from core.metrics_bodymask import ct_body_mask

# Calculates Root Mean Squared Deviation, optionally ignoring areas outside a body mask.
def _rmsd(a01, b01, mask01=None):
    if isinstance(a01, np.ndarray): a01 = torch.from_numpy(a01)
    if isinstance(b01, np.ndarray): b01 = torch.from_numpy(b01)
    a01, b01 = a01.float(), b01.float()
    diff2 = (a01 - b01) ** 2
    if mask01 is not None:
        m = mask01.float()
        num = diff2.mul(m).sum(dim=[1, 2, 3])
        den = m.sum(dim=[1, 2, 3]).clamp_min(1.0)
        val = torch.sqrt(num / den)
    else:
        val = torch.sqrt(diff2.mean(dim=[1, 2, 3]))
    return float(val.mean().detach().cpu())

# A small helper to get the final image from the diffusion process.
def _final_frame(t):
    if isinstance(t, (list, tuple)): return t[-1]
    if torch.is_tensor(t) and t.dim() == 5: return t[-1]
    return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ct_dose_train.json', help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None, help="Comma-separated GPU ids")
    parser.add_argument('--output_dir', type=str, default='experiments/sr_factor_analysis', help="Directory to save result plot")
    parser.add_argument('--sample_index', type=int, default=0, help="Index of the validation sample to use for the test")

    args = parser.parse_args()
    args.phase = 'val'
    args.enable_wandb = False
    args.debug = False
    args.log_wandb_ckpt = False
    args.log_eval = False
    args.log_infer = False
    
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    Logger.setup_logger(None, opt['path']['log'], 'sr_factor_test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info("Starting Super-Resolution Factor Analysis...")

    # Create the pre-trained model.
    diffusion = Model.create_model(opt)
    logger.info('Model created.')

    # Load the validation dataset.
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    logger.info(f'Validation dataset loaded with {len(val_set)} samples.')

    if args.sample_index >= len(val_set):
        raise ValueError(f"Sample index {args.sample_index} is out of bounds for validation set of size {len(val_set)}")
    
    test_sample = val_set[args.sample_index]
    # Add a batch dimension to the sample's tensors so it's ready for the model.
    for key, tensor in test_sample.items():
        if isinstance(tensor, torch.Tensor):
            test_sample[key] = tensor.unsqueeze(0)
    
    uid = test_sample.get('uid', f'sample_{args.sample_index}')
    logger.info(f"Using validation sample: UID {uid}")

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    diffusion.netG.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- EXPERIMENT SETUP ---
    hr_dose = test_sample['HR']
    ct_scan = test_sample['SR'][:, 1:2, :, :]
    
    # Normalize images to [0, 1] for metric calculation and create a body mask
    # to make sure error is only measured in relevant areas.
    hr_dose_01 = (hr_dose + 1) / 2
    ct_01 = (ct_scan + 1) / 2
    body_mask = ct_body_mask(ct_01)

    target_size = hr_dose.shape[-1]
    
    # Define the different super-resolution factors we want to test.
    sr_factors = [2, 3, 4, 5, 8, 10]
    results_rmsd = []
    
    for factor in sr_factors:
        low_res = target_size // factor
        logger.info(f"Testing SR Factor: {factor}x (downsampling to {low_res}x{low_res})")

        # 1. Manually create the low-resolution input by downsampling the ground truth.
        lr_dose = F.interpolate(hr_dose, size=(low_res, low_res), mode='bilinear', align_corners=False)
        
        # 2. Upsample it back to create the conditioning image.
        lr_up_dose = F.interpolate(lr_dose, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # 3. Combine the LR dose and the original CT to form the model's input.
        conditioning_tensor = torch.cat([lr_up_dose, ct_scan], dim=1)
        
        # 4. Feed this synthetically generated data to the model.
        inference_data = {'SR': conditioning_tensor, 'HR': hr_dose}
        diffusion.feed_data(inference_data)
        
        # 5. Run the standard inference process.
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()
        sr_final = _final_frame(visuals['SR'])
        
        # 6. Calculate the masked RMSD between the model's output and the ground truth.
        sr01 = (sr_final + 1) / 2
        device = sr01.device
        hr_dose_01_device = hr_dose_01.to(device)
        body_mask_device = body_mask.to(device)

        rmsd = _rmsd(sr01, hr_dose_01_device, body_mask_device)
        results_rmsd.append(rmsd)
        logger.info(f" -> Result: Masked RMSD = {rmsd:.5f}")

    # Plot how the error (RMSD) changes as the super-resolution task gets harder.
    plt.figure(figsize=(10, 6))
    plt.plot(sr_factors, results_rmsd, marker='o', linestyle='-', color='b')
    plt.title(f'Model Performance vs. Super-Resolution Factor (Sample UID {uid})')
    plt.xlabel('Super-Resolution Factor (e.g., 4x)')
    plt.ylabel('Masked RMSD (Lower is Better)')
    plt.grid(True, linestyle=':')
    plt.xticks(sr_factors)
    
    plot_path = os.path.join(args.output_dir, f'sr_factor_vs_rmsd_uid_{uid}.png')
    plt.savefig(plot_path)
    logger.info(f"Result plot saved to: {plot_path}")
    plt.show()