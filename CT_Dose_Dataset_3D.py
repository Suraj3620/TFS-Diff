import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from core.bspline_interpolator_3d import BSplineInterpolator3D

class CT_Dose_Dataset_3D(Dataset):
    def __init__(self, data_list, patch_size=32, input_size=16, phase='train'):
        self.data = data_list
        self.patch_size = patch_size
        self.input_size = input_size
        self.phase = phase
        self.bspline = BSplineInterpolator3D(method='quintic')
        self.air_threshold = -200.0 

        print(f"[*] 3D Dataset Initialized (Patch-Wise Norm).")
        print(f"    Mode: {phase}")
        print(f"    Patch: {input_size}^3 -> {patch_size}^3")

    def __len__(self):
        return len(self.data)

    def _get_crop_coords(self, vol_shape):
        d, h, w = vol_shape
        pd, ph, pw = self.patch_size, self.patch_size, self.patch_size
        
        # Safety check
        if d < pd or h < ph or w < pw:
            return 0, 0, 0

        if self.phase == 'train':
            z = np.random.randint(0, d - pd + 1)
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)
        else:
            # Center crop
            z, y, x = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
        return z, y, x

    def __getitem__(self, i):
        sample = self.data[i]
        vol_dose = sample['dose']
        vol_ct   = sample['ct']
        
        # 1. Extract Patch
        z, y, x = self._get_crop_coords(vol_dose.shape)
        pz = self.patch_size
        
        patch_dose = vol_dose[z:z+pz, y:y+pz, x:x+pz].astype(np.float32)
        patch_ct   = vol_ct[z:z+pz, y:y+pz, x:x+pz].astype(np.float32)

        # 2. Patch-Wise Normalization (The Fix)
        # Find local min/max to stretch contrast
        d_min, d_max = patch_dose.min(), patch_dose.max()
        
        # Handle empty/flat patches
        if (d_max - d_min) < 1e-9:
            # If patch is empty, force everything to -1
            dose_norm = np.full_like(patch_dose, -1.0)
            base_norm = np.full_like(patch_dose, -1.0)
            res_norm  = np.zeros_like(patch_dose)
        else:
            # Stretch to [-1, 1] based on LOCAL range
            dose_norm = (patch_dose - d_min) / (d_max - d_min)
            dose_norm = (dose_norm * 2.0) - 1.0
            
            # 3. Create Inputs from the NORMALIZED patch
            # This ensures degradation happens in the model's feature space
            t_hr = torch.from_numpy(dose_norm)[None, None]
            t_lr = F.interpolate(t_hr, size=(self.input_size,)*3, mode='trilinear', align_corners=True)
            lr_small = t_lr[0,0].numpy()
            
            # Upsample (Baseline)
            base_norm = self.bspline.resize_3d(lr_small, (pz, pz, pz))
            
            # 4. Calculate Residual
            # Target = HR - Baseline
            res_norm = dose_norm - base_norm

        # 5. Air Mask
        # Applied to residual to ensure we don't train on air noise
        air_mask = (patch_ct > self.air_threshold).astype(np.float32)
        res_norm = res_norm * air_mask

        # 6. CT Norm (Robust Global is fine for CT)
        ct_norm = np.clip(patch_ct, -1000, 1000)
        ct_norm = (ct_norm + 1000) / 2000.0 * 2.0 - 1.0

        # 7. Convert to Tensors
        # No extra scaling needed! Residuals in [-1, 1] space are naturally visible
        target_tensor = torch.from_numpy(res_norm).float().unsqueeze(0)
        input_base    = torch.from_numpy(base_norm).float().unsqueeze(0)
        input_ct      = torch.from_numpy(ct_norm).float().unsqueeze(0)
        
        cond_tensor = torch.cat([input_base, input_ct], dim=0)

        return {
            "RES": target_tensor, 
            "SR": cond_tensor,    
            "HR": torch.from_numpy(dose_norm).float().unsqueeze(0),
            "BASE": input_base,
            "MASK": torch.from_numpy(air_mask).float().unsqueeze(0),
            "LR": input_base # For visualization consistency
        }