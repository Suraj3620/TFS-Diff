import torch
from torch.utils.data import DataLoader

# Import your datasets
from .LRHR_Fusion_dataset import LRHRDataset
from .CT_Dose_Dataset_3D import CT_Dose_Dataset_3D
from .CT_Dose_Dataset_3D_Lazy import CT_Dose_Dataset_3D_Lazy  # <--- Make sure this is imported
from .Preprocessed_dataset import PreprocessedDataset
from .InMemory_dataset import InMemoryDataset 

def create_dataset(dataset_opt, phase):
    # 'mode' might be explicit, or we fallback to 'name'
    mode = dataset_opt.get('mode', dataset_opt.get('name'))
    
    if mode == 'CT_Dose_3D_Lazy':
        return CT_Dose_Dataset_3D_Lazy(
            csv_path=dataset_opt['dataroot'],
            patch_size=dataset_opt.get('patch_size', 32),
            input_size=dataset_opt.get('input_size', 16),
            phase=phase,
            # CRITICAL FIX: Actually read this param from JSON!
            subset_fraction=dataset_opt.get('subset_fraction', 1.0) 
        )

    elif mode == 'CT_Dose_3D':
        # Load .pt file
        data_list = torch.load(dataset_opt['dataroot'])
        return CT_Dose_Dataset_3D(
            data_list, 
            patch_size=dataset_opt.get('patch_size', 32),
            input_size=dataset_opt.get('input_size', 16),
            phase=phase
        )
        
    elif mode == 'InMemory':
        return InMemoryDataset(dataset_opt, phase)
    elif mode == 'Preprocessed':
        return PreprocessedDataset(dataset_opt, phase)
    elif mode == 'LRHR':
        return LRHRDataset(dataset_opt, phase)
    else:
        raise ValueError(f"Unsupported dataset mode: {mode}")

def create_dataloader(dataset, dataset_opt, phase):
    if phase == 'train':
        batch_size  = int(dataset_opt.get('batch_size', 1))
        use_shuffle = bool(dataset_opt.get('use_shuffle', True))
        num_workers = int(dataset_opt.get('num_workers', 0))
    else:
        batch_size  = 1
        use_shuffle = False
        num_workers = int(dataset_opt.get('num_workers', 0))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )