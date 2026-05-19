import os
import torch
import numpy as np
import csv
from tqdm import tqdm
import argparse

def preprocess_3d(index_csv, output_path):
    print(f"[*] Loading 3D volumes from {index_csv}...")
    
    data_list = []
    
    with open(index_csv, 'r') as f:
        reader = list(csv.DictReader(f))
        
        for row in tqdm(reader):
            try:
                # Load Dose
                dose_path = row['dose_path']
                dose_vol = np.load(dose_path, mmap_mode='r')
                # Ensure it's in memory (copy)
                dose_vol = np.array(dose_vol)

                # Load CT
                ct_fname = os.path.basename(dose_path)
                ct_path = os.path.join(row['ct_path'], ct_fname)
                ct_vol = np.load(ct_path, mmap_mode='r')
                ct_vol = np.array(ct_vol)
                
                # Check 3D
                if dose_vol.ndim != 3 or ct_vol.ndim != 3:
                    # Skip 2D files if they snuck in
                    continue
                    
                data_list.append({
                    'dose': dose_vol, # 100x100x100 numpy array
                    'ct': ct_vol,
                    'uid': row['uid']
                })
                
            except Exception as e:
                print(f"Error loading {dose_path}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data_list, output_path)
    print(f"[SUCCESS] Saved {len(data_list)} volumes to {output_path}")

if __name__ == '__main__':
    # Use the "All Cubes" index we generated earlier
    preprocess_3d('data/CT_Dose_Dataset/train/index.csv', 'data/CT_Dose_Dataset_InMemory/train_3d.pt')
    preprocess_3d('data/CT_Dose_Dataset/val/index.csv', 'data/CT_Dose_Dataset_InMemory/val_3d.pt')
    preprocess_3d('data/CT_Dose_Dataset/test/index.csv', 'data/CT_Dose_Dataset_InMemory/test_3d.pt')