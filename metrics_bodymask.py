import torch
import torch.nn.functional as F

def ct_body_mask(ct_in: torch.Tensor, thr_hu: float = -900.0):
    """
    Generates a body mask based on HU threshold.
    Handles both 2D (B, C, H, W) and 3D (B, C, D, H, W) inputs.
    """
    # ct_in is normalized to [-1, 1] with the mapping ((HU+1000)/2000)-1
    # invert normalization: HU = (val + 1)*2000 - 1000
    
    # Slice first channel (usually CT is 1 channel anyway)
    ct_data = ct_in[:, 0:1, ...] 
    
    hu = (ct_data + 1.0) * 2000.0 - 1000.0
    mask = (hu > thr_hu).float() # inside body

    # Dilate mask slightly to fill gaps
    if mask.dim() == 5: # 3D Case (B, C, D, H, W)
        mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    else: # 2D Case (B, C, H, W)
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        
    return mask

def masked_psnr(sr, hr, mask, eps=1e-8):
    # Works for both 2D and 3D as long as shapes match
    diff2 = (sr - hr)**2 * mask
    mse = diff2.sum() / (mask.sum() + eps)
    psnr = -10.0 * torch.log10(mse + eps)
    return psnr.item()