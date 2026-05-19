import torch
import matplotlib.pyplot as plt
from data.CT_Dose_Dataset_3D import CT_Dose_Dataset_3D

# Load data
data_list = torch.load('data/CT_Dose_Dataset_InMemory/train_3d.pt')

# Init Dataset
ds = CT_Dose_Dataset_3D(data_list, patch_size=32, input_size=16, phase='train')

# Get Sample
item = ds[0]

# Unpack (RES is scaled by 100, so divide back for viz)
res = item['RES'][0].numpy() / 100.0
base = item['BASE'][0].numpy()
hr = item['HR'][0].numpy()
mask = item['MASK'][0].numpy()

print(f"Residual Range: {res.min():.4f} to {res.max():.4f}")
print(f"Mask Mean: {mask.mean():.2f} (Should be < 1.0)")

# Plot Middle Slice (z=16)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(hr[16], cmap='gray'); ax[0].set_title('HR')
ax[1].imshow(base[16], cmap='gray'); ax[1].set_title('B-Spline Base')
ax[2].imshow(mask[16], cmap='gray'); ax[2].set_title('Air Mask')
ax[3].imshow(res[16], cmap='bwr'); ax[3].set_title('Diff to Water')
plt.savefig("check_3d_loader.png")
print("Saved check_3d_loader.png")