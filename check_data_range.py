import torch
import numpy as np

# Load the training set we just created
data_path = 'data/CT_Dose_Dataset_InMemory/train_set.pt'
print(f"[*] Loading {data_path}...")
dataset = torch.load(data_path)

max_vals = []
min_vals = []

# Check a random subset
indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)

print(f"[*] Inspecting {len(indices)} random samples...")

for i in indices:
    # HR is [1, H, W]
    hr = dataset[i]['HR'].numpy()
    max_vals.append(hr.max())
    min_vals.append(hr.min())

global_max = np.max(max_vals)
global_min = np.min(min_vals)
avg_max = np.mean(max_vals)

print("-" * 40)
print(f"Data Statistics (Normalized):")
print(f"  Global Max found: {global_max:.6f}")
print(f"  Global Min found: {global_min:.6f}")
print(f"  Average Max per slice: {avg_max:.6f}")
print("-" * 40)

if global_max < -0.9:
    print("⚠️  CRITICAL WARNING: Data is compressed too much!")
    print("    Your max pixel value is near -1.0.")
    print("    This means 'global_max_dose=85.0' was WAY too high.")
    print("    The model sees nothing but black.")
elif global_max > 0.9:
    print("✅  Data range looks healthy (uses the full -1 to 1 spectrum).")
else:
    print("ℹ️  Data range is somewhat compressed, but visible.")