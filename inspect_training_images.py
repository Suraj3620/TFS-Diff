import torch
import cv2
import numpy as np
import os

# Load the training set
data_path = 'data/CT_Dose_Dataset_InMemory/train_set.pt'
print(f"[*] Loading {data_path}...")
dataset = torch.load(data_path)

# Get a sample
idx = 10  # Arbitrary index
sample = dataset[idx]

# Extract tensors
hr_tensor = sample['HR']          # [1, H, W]
lr_tensor = sample['SR'][0]       # [H, W] (First channel of SR input)
ct_tensor = sample['SR'][1]       # [H, W] (Second channel of SR input)

print(f"UID: {sample['uid']}")
print(f"HR Range: {hr_tensor.min():.4f} to {hr_tensor.max():.4f}")
print(f"LR Range: {lr_tensor.min():.4f} to {lr_tensor.max():.4f}")

# Convert to images for saving
def to_img(t):
    x = t.numpy().squeeze()
    # Map -1..1 to 0..255
    x = (x + 1) / 2 * 255
    return x.astype(np.uint8)

img_hr = to_img(hr_tensor)
img_lr = to_img(lr_tensor)
img_ct = to_img(ct_tensor)

# Save
cv2.imwrite("debug_train_HR.png", img_hr)
cv2.imwrite("debug_train_LR.png", img_lr)
cv2.imwrite("debug_train_CT.png", img_ct)

print("[*] Saved debug images: debug_train_HR.png, debug_train_LR.png, debug_train_CT.png")
print("--> PLEASE CHECK THESE IMAGES. Do they look like valid anatomy/dose?")