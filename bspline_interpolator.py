import numpy as np
import scipy.interpolate as si
import torch

class BSplineInterpolator:
    def __init__(self, k=5):
        """
        Tensor Product B-Spline Interpolator.
        k=5 (5 knots/order) as requested.
        """
        self.k = k

    def resize_2d(self, img_lr, target_shape):
        """
        img_lr: Numpy array (H, W)
        target_shape: Tuple (H_new, W_new)
        """
        h, w = img_lr.shape
        h_new, w_new = target_shape

        # 1. Define the Input Grid
        # We use the exact integer coordinates of the input pixels.
        # This corresponds to align_corners=True.
        x = np.arange(h)
        y = np.arange(w)

        # 2. Create the Spline Model
        # s=0 ensures we pass exactly through the input points (Interpolation, not Approximation)
        spline_model = si.RectBivariateSpline(x, y, img_lr, kx=self.k, ky=self.k, s=0)

        # 3. Define the Output Grid
        # We map the output range [0, h_new-1] exactly to input range [0, h-1]
        x_new = np.linspace(0, h - 1, h_new)
        y_new = np.linspace(0, w - 1, w_new)

        # 4. Evaluate
        img_hr = spline_model(x_new, y_new)
        
        # 5. Clip to ensure numerical stability (splines can slightly overshoot min/max)
        img_hr = np.clip(img_hr, img_lr.min(), img_lr.max())
        
        return img_hr.astype(np.float32)