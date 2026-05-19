import os, argparse, torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import data as Data
import core.logger as Logger
from models.unet_baseline import UNetBaseline

# --------- helpers to resolve keys and build inputs ---------
# A flexible helper to find a tensor in the batch dictionary using a list of possible keys.
def _pick(batch, candidates, what):
    for k in candidates:
        if k in batch:
            return batch[k]
    raise KeyError(f"Missing '{what}'. Available keys: {list(batch.keys())}")

# Checks if a CT scan tensor is present in the provided data batch.
def detect_with_ct(sample_batch):
    return any(k in sample_batch for k in ["CT","ct","CT_m11","CT_M11","ct_m11","COND","cond","condition","conditioning"])

# Assembles the model's input (x) and the ground truth target (hr) from a data batch.
def get_inputs(batch, with_ct: bool):
    """
    Returns:
      x: (B,in_ch,H,W) where in_ch=2 if with_ct else 1
      hr: (B,1,H,W)
    """
    lr = _pick(batch, ["LR","lr","LR_UP","LR_up","lr_up","LRUP","LR_BICUBIC","LR_UPSAMPLED"], "LR (upsampled LR)")
    hr = _pick(batch, ["HR","hr","GT","gt","target","Target","dose_hr","HR_m11","hr_m11"], "HR/target")

    # Ensures tensors have the expected 4D shape (B, C, H, W).
    if lr.dim() == 3:  lr  = lr.unsqueeze(1)
    if hr.dim() == 3:  hr  = hr.unsqueeze(1)

    # If conditioning on a CT scan, it concatenates the CT and LR tensors.
    if with_ct:
        ct = _pick(batch, ["CT","ct","CT_m11","CT_M11","ct_m11","COND","cond","condition","conditioning"], "CT/conditioning")
        if ct.dim() == 3: ct = ct.unsqueeze(1)
        x = torch.cat([ct, lr], dim=1)
    else:
        x = lr
    return x, hr

# ------------------------------ main ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config",default="config/ct_dose_train.json")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--out",default="experiments/unet_baseline")

    # These arguments are included to maintain compatibility with the project's standard Logger.
    ap.add_argument("-gpu","--gpu_ids", type=str, default=None, help='Comma-separated GPU ids (e.g. "0")')
    ap.add_argument("--gpu", type=str, default=None, help='Alias for single GPU id')
    ap.add_argument("-debug","-d", action="store_true")
    ap.add_argument("-enable_wandb", action="store_true")
    ap.add_argument("-log_wandb_ckpt", action="store_true")
    ap.add_argument("-log_eval", action="store_true")
    ap.add_argument("-log_infer", action="store_true")

    args = ap.parse_args()
    if args.gpu_ids is None and args.gpu is not None:
        args.gpu_ids = args.gpu
    args.phase = 'train'

    # Uses the project's existing configuration parser to set up paths and environment.
    opt = Logger.parse(args); opt = Logger.dict_to_nonedict(opt)

    # Leverages the project's data loading framework to create the datasets and dataloaders.
    train_set = Data.create_dataset(opt['datasets']['train'], 'train')
    val_set   = Data.create_dataset(opt['datasets']['val'],   'val')
    train_loader = Data.create_dataloader(train_set, opt['datasets']['train'], 'train')
    val_loader   = Data.create_dataloader(val_set,   opt['datasets']['val'],   'val')

    # This step inspects the first batch to dynamically determine
    # whether the model should have 1 input channel (LR only) or 2 (LR + CT).
    first_batch = next(iter(train_loader))
    WITH_CT = detect_with_ct(first_batch)
    in_ch = 2 if WITH_CT else 1
    print(f"[unet-baseline] with_ct={WITH_CT} -> in_ch={in_ch}")

    # Standard PyTorch setup: model, optimizer, loss function, and a GradScaler for mixed precision.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNetBaseline(in_ch=in_ch, base=64).to(device)
    opti = torch.optim.AdamW(net.parameters(), lr=float(args.lr), betas=(0.9,0.99))
    crit = nn.L1Loss()
    scaler = GradScaler()

    os.makedirs(args.out, exist_ok=True)

    best_val = 1e9
    # The main training loop over the specified number of epochs.
    for ep in range(1, int(args.epochs)+1):
        net.train(); tbar = tqdm(train_loader, desc=f"train ep{ep}")
        tloss=0.0
        # The inner loop iterates through batches of the training data.
        for batch in tbar:
            x, hr = get_inputs(batch, WITH_CT)
            x, hr = x.to(device), hr.to(device)
            opti.zero_grad(set_to_none=True)
            # `autocast` enables Automatic Mixed Precision (AMP) for faster training.
            with autocast():
                sr = net(x)
                loss = crit(sr, hr)
            # The GradScaler handles the backward pass and optimizer step for AMP.
            scaler.scale(loss).backward()
            scaler.step(opti); scaler.update()
            tloss += loss.item()
            tbar.set_postfix(loss=f"{loss.item():.4f}")
        tloss /= max(1,len(train_loader))

        # After each epoch, the model is evaluated on the validation set.
        net.eval(); vloss=0.0
        with torch.no_grad():
            for batch in val_loader:
                x, hr = get_inputs(batch, WITH_CT)
                x, hr = x.to(device), hr.to(device)
                with autocast():
                    sr = net(x)
                    vloss += crit(sr, hr).item()
        vloss /= max(1,len(val_loader))
        print(f"[ep {ep}] train L1={tloss:.4f} | val L1={vloss:.4f}")

        # The model's state is saved only if it achieves a new best validation loss.
        if vloss < best_val:
            best_val = vloss
            ckpt = os.path.join(args.out, f"unet_baseline_best.pth")
            torch.save({"model": net.state_dict(), "in_ch": in_ch}, ckpt)
            print("[save]", ckpt)