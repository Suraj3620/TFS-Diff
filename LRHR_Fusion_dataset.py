import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from core.bspline_interpolator import BSplineInterpolator

def _resize_hw(img, hw_out):
    if img.shape == tuple(hw_out): return img.astype(np.float32, copy=False)
    t = torch.from_numpy(img.astype(np.float32))[None,None]
    # FIX: Align corners for consistency
    r = F.interpolate(t, size=hw_out, mode='bilinear', align_corners=True)
    return r[0,0].numpy()

class LRHRDataset(Dataset):
    def __init__(self, opt, phase):
        super().__init__()
        self.root = opt['dataroot']
        self.l_res = int(opt.get('l_resolution', 25))
        self.r_res = int(opt.get('r_resolution', 100))
        self.use_shuffle = bool(opt.get('use_shuffle', True)) and phase == 'train'
        
        self.bspline = BSplineInterpolator(k=5)

        with open(os.path.join(self.root, "index.csv"), "r") as f:
            self.rows = list(csv.DictReader(f))

        if self.use_shuffle:
            random.shuffle(self.rows)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            dose_path = r["dose_path"]
            ct_path = os.path.join(r["ct_path"], os.path.basename(dose_path))
            
            dose_raw = np.load(dose_path, mmap_mode="r")
            if not os.path.isfile(ct_path):
                ct_raw = np.zeros_like(dose_raw)
            else:
                ct_raw = np.load(ct_path, mmap_mode="r")

            target_slice_idx = int(r["slice"])
            
            if dose_raw.ndim == 2: dose = dose_raw
            elif dose_raw.ndim == 3:
                safe_idx = min(target_slice_idx, dose_raw.shape[-1] - 1)
                dose = dose_raw[..., safe_idx]
            else: raise ValueError("Bad shape")

            if ct_raw.ndim == 2: ct = ct_raw
            elif ct_raw.ndim == 3:
                safe_idx = min(target_slice_idx, ct_raw.shape[-1] - 1)
                ct = ct_raw[..., safe_idx]
            else: ct = np.zeros_like(dose)

        except Exception:
            return self.__getitem__((i + 1) % len(self.rows))

        if dose.shape != (self.r_res, self.r_res):
            dose = _resize_hw(dose, (self.r_res, self.r_res))
        if ct.shape != (self.r_res, self.r_res):
            ct = _resize_hw(ct, (self.r_res, self.r_res))

        # --- PIPELINE UPDATE ---
        
        # 1. Downsample (Degradation)
        # CRITICAL FIX: align_corners=True to match B-Spline coordinate system
        t_hr = torch.from_numpy(dose.astype(np.float32))[None,None]
        t_lr = F.interpolate(t_hr, size=(self.l_res, self.l_res), mode='bilinear', align_corners=True)
        dose_lr_small = t_lr[0,0].numpy()

        # 2. B-Spline Upsampling (Baseline)
        dose_lr_up = self.bspline.resize_2d(dose_lr_small, (self.r_res, self.r_res))

        # 3. Calculate Residuals
        # Range of dose is small, so we calculate residual on raw values first, then normalize?
        # No, standard practice: Normalize images first, then subtract.
        
        d_min, d_max = float(dose.min()), float(dose.max())
        
        if (d_max - d_min) < 1e-6:
            hr_norm = np.zeros_like(dose) - 1.0
            lr_norm = np.zeros_like(dose_lr_up) - 1.0
            res_norm = np.zeros_like(dose)
        else:
            # Normalize HR
            hr_norm = (dose - d_min) / (d_max - d_min)
            hr_norm = (hr_norm * 2.0) - 1.0
            
            # Normalize LR (using HR stats to stay consistent)
            lr_norm = (dose_lr_up - d_min) / (d_max - d_min)
            lr_norm = (lr_norm * 2.0) - 1.0
            
            # Residual
            res_norm = hr_norm - lr_norm

        # CT Norm
        c_lo, c_hi = np.percentile(ct, 1), np.percentile(ct, 99)
        if c_hi <= c_lo: ct_norm = np.zeros_like(ct) - 1.0
        else:
            ct_norm = np.clip(ct, c_lo, c_hi)
            ct_norm = (ct_norm - c_lo) / (c_hi - c_lo)
            ct_norm = (ct_norm * 2.0) - 1.0

        HR = torch.from_numpy(hr_norm).float().unsqueeze(0)
        RES = torch.from_numpy(res_norm).float().unsqueeze(0)
        LR = torch.from_numpy(lr_norm).float().unsqueeze(0)
        CT = torch.from_numpy(ct_norm).float().unsqueeze(0)
        SR = torch.cat([LR, CT], dim=0)

        return {"HR": HR, "RES": RES, "SR": SR, "LR": LR, "uid": r["uid"], "slice": target_slice_idx}