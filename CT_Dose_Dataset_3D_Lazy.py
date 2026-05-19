import os
import torch
import numpy as np
import csv
from torch.utils.data import Dataset
import torch.nn.functional as F
from core.bspline_interpolator_3d import BSplineInterpolator3D

class CT_Dose_Dataset_3D_Lazy(Dataset):
    def __init__(self, csv_path, patch_size=32, input_size=16, phase='train', subset_fraction=1.0):
        self.patch_size = patch_size
        self.input_size = input_size
        self.phase = phase
        self.bspline = BSplineInterpolator3D(method='quintic')
        self.air_threshold = -200.0 
        
        # --- RESIDUAL SCALING FACTOR ---
        # We scale residuals by 50.
        # A typical dose error of 0.02 becomes 1.0 (Unit Variance).
        # This ensures the Diffusion Model sees a strong signal.
        self.RESIDUAL_SCALE = 50.0 

        self.data_index = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = list(csv.DictReader(f))
                self.data_index = reader
        else:
            print(f"[ERROR] CSV not found: {csv_path}")

        # Subsampling Logic
        if subset_fraction < 1.0 and len(self.data_index) > 0:
            total = len(self.data_index)
            keep = int(max(1, total * subset_fraction))
            rng = np.random.RandomState(42)
            indices = rng.choice(total, keep, replace=False)
            self.data_index = [self.data_index[i] for i in indices]
            print(f"[*] Lazy Dataset ({phase}): Subsampled {keep}/{total}")
        else:
            print(f"[*] Lazy Dataset ({phase}): Loaded full {len(self.data_index)}")

    def __len__(self):
        return len(self.data_index)

    def _get_crop_coords(self, vol_shape):
        d, h, w = vol_shape
        pd, ph, pw = self.patch_size, self.patch_size, self.patch_size
        
        if d < pd: z = 0
        elif self.phase == 'train': z = np.random.randint(0, d - pd + 1)
        else: z = (d - pd) // 2

        if h < ph: y = 0
        elif self.phase == 'train': y = np.random.randint(0, h - ph + 1)
        else: y = (h - ph) // 2

        if w < pw: x = 0
        elif self.phase == 'train': x = np.random.randint(0, w - pw + 1)
        else: x = (w - pw) // 2
        return z, y, x

    def __getitem__(self, i):
        row = self.data_index[i]
        try:
            dose_path = row['dose_path']
            ct_path = os.path.join(row['ct_path'], os.path.basename(dose_path))
            vol_dose = np.load(dose_path, mmap_mode='r')
            vol_ct = np.load(ct_path, mmap_mode='r')
        except Exception as e:
            return self.__getitem__((i + 1) % len(self.data_index))

        # --- Hard Example Mining ---
        best_patch = None
        best_err = -1.0
        # Train: 10 attempts to find beam. Val: 1 attempt (center)
        attempts = 10 if self.phase == 'train' else 1
        
        for _ in range(attempts):
            z, y, x = self._get_crop_coords(vol_dose.shape)
            pz = self.patch_size
            
            p_dose = np.array(vol_dose[z:z+pz, y:y+pz, x:x+pz], dtype=np.float32)
            
            # Skip Empty Air
            if (p_dose.max() - p_dose.min()) < 1e-9: continue
            
            # 1. Normalize [-1, 1]
            d_min, d_max = p_dose.min(), p_dose.max()
            dose_01 = (p_dose - d_min) / (d_max - d_min)
            
            # 2. Generate Baseline
            t_hr = torch.from_numpy(dose_01)[None, None]
            t_lr = F.interpolate(t_hr, size=(self.input_size,)*3, mode='trilinear', align_corners=True)
            lr_s = t_lr[0,0].numpy()
            base_01 = self.bspline.resize_3d(lr_s, (pz, pz, pz))
            
            # 3. Check Residual
            err = np.max(np.abs(dose_01 - base_01))
            if err > best_err:
                best_err = err
                p_ct = np.array(vol_ct[z:z+pz, y:y+pz, x:x+pz], dtype=np.float32)
                best_patch = (dose_01, base_01, p_ct)
            
            if best_err > 0.02: break
            
        if best_patch is None:
            # Fallback
            z, y, x = self._get_crop_coords(vol_dose.shape)
            pz = self.patch_size
            dose_01 = np.zeros((pz,pz,pz), dtype=np.float32)
            base_01 = np.zeros((pz,pz,pz), dtype=np.float32)
            p_ct = np.zeros((pz,pz,pz), dtype=np.float32)
        else:
            dose_01, base_01, p_ct = best_patch

        # --- Final Tensors ---
        dose_norm = (dose_01 * 2.0) - 1.0
        base_norm = (base_01 * 2.0) - 1.0
        
        # EXACT True Residual. No scaling. Range is naturally around [-1.5, 1.5]
        res_norm = dose_norm - base_norm
        
        ct_norm = np.clip(p_ct, -1000, 1000)
        ct_norm = (ct_norm + 1000) / 2000.0 * 2.0 - 1.0
        
        air_mask = (p_ct > self.air_threshold).astype(np.float32)
        res_norm = res_norm * air_mask

        return {
            "RES": torch.from_numpy(res_norm).float().unsqueeze(0),
            "SR": torch.cat([
                torch.from_numpy(base_norm).float().unsqueeze(0),
                torch.from_numpy(ct_norm).float().unsqueeze(0)
            ], dim=0),
            "HR": torch.from_numpy(dose_norm).float().unsqueeze(0),
            "LR": torch.from_numpy(base_norm).float().unsqueeze(0)
        }