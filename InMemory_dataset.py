import os
import torch
from torch.utils.data import Dataset

class InMemoryDataset(Dataset):
    def __init__(self, opt, phase):
        super().__init__()
        self.data_file_path = opt['dataroot']
        
        print(f"[*] Loading entire dataset into memory from '{self.data_file_path}'...")
        self.data = torch.load(self.data_file_path, weights_only=False)
        print(f"[*] In-Memory Dataset loaded with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]