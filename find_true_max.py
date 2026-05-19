import os
import numpy as np
import csv
from tqdm import tqdm

CSV_PATH = "data/CT_Dose_Dataset/train/index.csv"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found")
        return

    print(f"[*] Scanning {CSV_PATH}...")
    global_max = -float('inf')
    
    with open(CSV_PATH, 'r') as f:
        reader = list(csv.DictReader(f))
        for row in tqdm(reader):
            try:
                data = np.load(row['dose_path'], mmap_mode='r')
                if data.ndim == 3:
                    sl = int(row['slice'])
                    if sl < data.shape[-1]: val = data[..., sl].max()
                    else: continue
                else:
                    val = data.max()
                if val > global_max: global_max = val
            except: pass

    print(f"\n[RESULT] True Max in Training Set: {global_max:.6f}")

if __name__ == "__main__":
    main()