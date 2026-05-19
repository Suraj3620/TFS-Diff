import numpy as np
import scipy.interpolate as si
import torch

class BSplineInterpolator3D:
    def __init__(self, method='quintic'):
        """
        3D Tensor Product B-Spline Interpolator.
        
        Args:
            method (str): 'quintic' corresponds to k=5 (5 knots) B-Splines.
                          'cubic' corresponds to k=3.
                          'linear' corresponds to k=1.
        """
        self.method = method

    def resize_3d(self, vol_lr, target_shape):
        """
        Resizes a 3D volume using RegularGridInterpolator (Tensor Product Spline).
        
        Args:
            vol_lr: Numpy array (D, H, W) -> Low Resolution Input
            target_shape: Tuple (D_new, H_new, W_new) -> High Resolution Output
            
        Returns:
            vol_hr: Numpy array (D_new, H_new, W_new)
        """
        # Ensure input is float for precision
        vol_lr = vol_lr.astype(np.float32)
        d, h, w = vol_lr.shape
        d_new, h_new, w_new = target_shape

        # 1. Define Input Grid coordinates (exact integer indices)
        # This aligns the corners: index 0 is 0.0, index N-1 is (N-1).0
        z = np.arange(d)
        y = np.arange(h)
        x = np.arange(w)

        # 2. Create the Interpolator Function
        # RegularGridInterpolator performs tensor-product interpolation in n-dimensions.
        # bounds_error=False and fill_value=None allow minor float discrepancies at edges safely.
        interpolator = si.RegularGridInterpolator(
            (z, y, x), 
            vol_lr, 
            method=self.method, 
            bounds_error=False, 
            fill_value=None
        )

        # 3. Define Output Grid coordinates
        # We map [0, d_new-1] exactly to [0, d-1]
        z_new = np.linspace(0, d - 1, d_new)
        y_new = np.linspace(0, h - 1, h_new)
        x_new = np.linspace(0, w - 1, w_new)

        # 4. Create the meshgrid for query points
        # 'indexing="ij"' ensures matrix indexing (D, H, W) order
        gz, gy, gx = np.meshgrid(z_new, y_new, x_new, indexing='ij')
        
        # Flatten to list of points: (N, 3)
        points = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=1)

        # 5. Interpolate
        vol_hr_flat = interpolator(points)
        
        # Reshape back to volume
        vol_hr = vol_hr_flat.reshape((d_new, h_new, w_new))

        # 6. Clip to input range to prevent B-Spline overshoot (ringing) blowing up values
        vol_hr = np.clip(vol_hr, vol_lr.min(), vol_lr.max())

        return vol_hr.astype(np.float32)

    def resize_batch_tensor(self, tensor_lr, target_shape):
        """
        Helper to handle PyTorch Tensors [B, C, D, H, W]
        """
        device = tensor_lr.device
        np_lr = tensor_lr.cpu().numpy()
        
        B, C, D, H, W = np_lr.shape
        D_new, H_new, W_new = target_shape
        
        np_hr = np.zeros((B, C, D_new, H_new, W_new), dtype=np.float32)

        for b in range(B):
            for c in range(C):
                np_hr[b, c] = self.resize_3d(np_lr[b, c], (D_new, H_new, W_new))
                
        return torch.from_numpy(np_hr).to(device)