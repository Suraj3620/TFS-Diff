import os
import csv
from tqdm import tqdm
import torch
from data.LRHR_Fusion_dataset import LRHRDataset

def run_preprocessing(original_dataroot, output_dataroot):
    """
    Reads the original dataset, processes each item, and saves the resulting
    tensors to a new directory. Creates a new index.csv for the fast dataset.
    """
    print(f"[*] Starting preprocessing...")
    print(f"[*] Original data root: {original_dataroot}")
    print(f"[*] Output data root:   {output_dataroot}")

    os.makedirs(output_dataroot, exist_ok=True)

    # Use a dummy options dict to initialize the original dataset
    opt = {'dataroot': original_dataroot}
    original_dataset = LRHRDataset(opt, phase='train') # Phase doesn't matter here

    new_index_data = []
    
    for i in tqdm(range(len(original_dataset)), desc="Preprocessing samples"):
        sample = original_dataset[i]
        
        # Define a unique filename for the preprocessed data
        uid = sample['uid']
        sl = sample['slice']
        output_filename = f"sample_{uid}_slice_{sl}.pt"
        output_path = os.path.join(output_dataroot, output_filename)
        
        # Save the tensor dictionary to a file
        torch.save(sample, output_path)
        
        # Add a row to the new index file
        new_index_data.append({
            'relative_path': output_filename,
            'uid': uid,
            'slice': sl
        })

    # Write the new, fast index file
    new_csv_path = os.path.join(output_dataroot, 'index_preprocessed.csv')
    with open(new_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['relative_path', 'uid', 'slice'])
        writer.writeheader()
        writer.writerows(new_index_data)

    print("\n" + "="*50)
    print(f"[SUCCESS] Preprocessing complete.")
    print(f"[*] New dataset saved to: {output_dataroot}")
    print(f"[*] New index file is at: {new_csv_path}")
    print("="*50)


if __name__ == '__main__':
    # Run this for both the training and validation sets
    
    print("--- PREPROCESSING TRAINING SET ---")
    run_preprocessing(
        original_dataroot='/home/sdesai/THESIS/TFS-Diff-main/data/CT_Dose_Dataset/train',
        output_dataroot='/home/sdesai/THESIS/TFS-Diff-main/data/CT_Dose_Dataset_Preprocessed/train'
    )
    
    print("\n--- PREPROCESSING VALIDATION SET ---")
    run_preprocessing(
        original_dataroot='/home/sdesai/THESIS/TFS-Diff-main/data/CT_Dose_Dataset/val',
        output_dataroot='/home/sdesai/THESIS/TFS-Diff-main/data/CT_Dose_Dataset_Preprocessed/val'
    )