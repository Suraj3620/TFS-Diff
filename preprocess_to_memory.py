import os
import torch
import argparse
from data.LRHR_Fusion_dataset import LRHRDataset
from tqdm import tqdm

def preprocess_subset(phase, csv_root, output_path, global_max=100.0):
    print(f"\n[*] Processing {phase} set...")
    print(f"    Root: {csv_root}")
    print(f"    Save to: {output_path}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize Dataset (This triggers the csv reading logic)
    # We pass 'global_max_dose' to ensure consistent normalization across all sets
    opt = {
        'dataroot': csv_root, 
        'use_shuffle': False,
        'l_resolution': 25,
        'r_resolution': 100,
        'global_max_dose': global_max 
    }
    
    try:
        dataset = LRHRDataset(opt, phase=phase)
    except FileNotFoundError as e:
        print(f"[SKIP] Could not load dataset for {phase}: {e}")
        return

    # Cache list
    cached_data = []
    for i in tqdm(range(len(dataset)), desc=f"Caching {phase}"):
        try:
            sample = dataset[i]
            cached_data.append(sample)
        except Exception as e:
            print(f"    [Warning] Failed to load sample {i}: {e}")

    # Save
    torch.save(cached_data, output_path)
    print(f"[SUCCESS] Saved {len(cached_data)} samples to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='data/CT_Dose_Dataset', help="Folder containing train/val/test subfolders")
    parser.add_argument('--save_root', default='data/CT_Dose_Dataset_InMemory', help="Where to save .pt files")
    parser.add_argument('--global_max_dose', type=float, default=100.0, help="Global max dose for normalization")
    args = parser.parse_args()

    # Process Train
    preprocess_subset('train', 
                      os.path.join(args.raw_root, 'train'), 
                      os.path.join(args.save_root, 'train_set.pt'),
                      args.global_max_dose)

    # Process Val
    preprocess_subset('val', 
                      os.path.join(args.raw_root, 'val'), 
                      os.path.join(args.save_root, 'val_set.pt'),
                      args.global_max_dose)
                      
    # Process Test (Optional, but good to have)
    preprocess_subset('test', 
                      os.path.join(args.raw_root, 'test'), 
                      os.path.join(args.save_root, 'test_set.pt'),
                      args.global_max_dose)