import os
import csv
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, opt, phase):
        super().__init__()
        self.root = opt['dataroot']
        index_csv = os.path.join(self.root, "index_preprocessed.csv")
        
        with open(index_csv, "r") as f:
            self.rows = list(csv.DictReader(f))
            
        print(f"[*] Fast PreprocessedDataset loaded with {len(self.rows)} samples.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        file_path = os.path.join(self.root, row['relative_path'])
        return torch.load(file_path)