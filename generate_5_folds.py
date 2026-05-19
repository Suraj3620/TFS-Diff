import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

def main():
    master_csv = "data/CT_Dose_Dataset/master_index.csv"
    output_root = "data/CT_Dose_Dataset/folds"
    
    if not os.path.exists(master_csv):
        print(f"[!] Error: Could not find {master_csv}")
        return

    df = pd.read_csv(master_csv)
    
    # 1. Get unique patients (UIDs) to ensure strict patient-level splitting
    unique_uids = df['uid'].unique()
    print(f"[*] Found {len(unique_uids)} unique patients.")

    # 2. Setup 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    os.makedirs(output_root, exist_ok=True)

    # 3. Create the splits
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_uids), 1):
        train_uids = unique_uids[train_idx]
        val_uids = unique_uids[val_idx]
        
        train_df = df[df['uid'].isin(train_uids)]
        val_df = df[df['uid'].isin(val_uids)]
        
        fold_dir = os.path.join(output_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(fold_dir, "train_index.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val_index.csv"), index=False)
        
        print(f"[*] Fold {fold}:")
        print(f"    Train: {len(train_uids)} patients ({len(train_df)} slices)")
        print(f"    Val:   {len(val_uids)} patients ({len(val_df)} slices)")

if __name__ == "__main__":
    main()