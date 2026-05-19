import pandas as pd
import numpy as np
import os

csv_path = "data/CT_Dose_Dataset/train/index.csv"
if not os.path.exists(csv_path):
    print("Index not found.")
    exit()

df = pd.read_csv(csv_path)
print(f"Checking {len(df)} slices in {csv_path}...")

max_vals = []
for _, row in df.iterrows():
    try:
        arr = np.load(row['dose_path'], mmap_mode='r')
        if arr.ndim == 3:
            sl = int(row['slice'])
            if sl < arr.shape[-1]: val = arr[..., sl].max()
            else: continue
        else:
            val = arr.max()
        max_vals.append(val)
    except: pass

max_vals = np.array(max_vals)
print(f"Count: {len(max_vals)}")
print(f"Max Dose in Dataset: {max_vals.max():.6f}")
print(f"Mean Max Dose:       {max_vals.mean():.6f}")
print(f"Slices near zero (< 0.001): {np.sum(max_vals < 0.001)} ({np.sum(max_vals < 0.001)/len(max_vals)*100:.1f}%)")