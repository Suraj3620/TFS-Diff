import os, argparse, torch, pandas as pd
from tqdm import tqdm
import numpy as np
import lpips
import data as Data
import core.logger as Logger
from models.unet_baseline import UNetBaseline
from core.metrics_bodymask import ct_body_mask

def _pick(batch, candidates, what):
    for k in candidates:
        if k in batch:
            return batch[k]
    raise KeyError(f"Missing '{what}'. Available keys: {list(batch.keys())}")

def _rmsd(a,b,mask=None):
    a,b=a.float(),b.float()
    diff2=(a-b)**2
    mse=(diff2*mask).sum()/mask.sum().clamp_min(1.0) if mask is not None else diff2.mean()
    return torch.sqrt(mse.clamp_min(1e-12)).item()

def _psnr(a,b,mask=None):
    a,b=a.float(),b.float()
    diff2=(a-b)**2
    mse=(diff2*mask).sum()/mask.sum().clamp_min(1.0) if mask is not None else diff2.mean()
    return -10.0*torch.log10(mse.clamp_min(1e-12)).item()

def detect_with_ct(sample_batch):  # same logic as trainer
    return any(k in sample_batch for k in ["CT","ct","CT_m11","CT_M11","ct_m11","COND","cond","condition","conditioning"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config",default="config/ct_dose_train.json")
    ap.add_argument("--ckpt",required=True,help="experiments/unet_baseline/unet_baseline_best.pth")
    ap.add_argument("-gpu","--gpu_ids", type=str, default=None)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--outcsv",default="experiments/unet_baseline_eval.csv")
    # logger flags (optional)
    ap.add_argument("-debug","-d", action="store_true")
    ap.add_argument("-enable_wandb", action="store_true")
    ap.add_argument("-log_wandb_ckpt", action="store_true")
    ap.add_argument("-log_eval", action="store_true")
    ap.add_argument("-log_infer", action="store_true")

    args = ap.parse_args()
    if args.gpu_ids is None and args.gpu is not None:
        args.gpu_ids = args.gpu
    args.phase='val'

    opt = Logger.parse(args); opt = Logger.dict_to_nonedict(opt)
    val_set = Data.create_dataset(opt['datasets']['val'], 'val')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')

    first_batch = next(iter(val_loader))
    WITH_CT = detect_with_ct(first_batch)
    # load model and honor saved in_ch if present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)
    in_ch = state.get("in_ch", 2 if WITH_CT else 1)
    print(f"[unet-baseline-eval] with_ct={WITH_CT} -> in_ch={in_ch}")
    net = UNetBaseline(in_ch=in_ch, base=64).to(device)
    net.load_state_dict(state["model"], strict=True)
    net.eval()

    lpips_model = lpips.LPIPS(net='vgg').to(device)   # same as the diffusion logs
    lpips_model.eval()

    results=[]
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="U-Net baseline eval"):
            lr = _pick(batch, ["LR","lr","LR_UP","LR_up","lr_up","LRUP","LR_BICUBIC","LR_UPSAMPLED"], "LR (upsampled LR)")
            hr_m11 = _pick(batch, ["HR","hr","GT","gt","target","Target","dose_hr","HR_m11","hr_m11"], "HR/target")
            if lr.dim()==3: lr = lr.unsqueeze(1)
            if hr_m11.dim()==3: hr_m11 = hr_m11.unsqueeze(1)

            if in_ch == 2:
                ct = _pick(batch, ["CT","ct","CT_m11","CT_M11","ct_m11","COND","cond","condition","conditioning"], "CT/conditioning")
                if ct.dim()==3: ct = ct.unsqueeze(1)
                x = torch.cat([ct.to(device), lr.to(device)], dim=1)
                mask = ct_body_mask(ct.cpu())
            else:
                x = lr.to(device)
                # if CT not present, make a full-ones mask (i.e., global metrics)
                mask = ct_body_mask(lr.cpu()) if "CT" in batch else None

            sr_m11 = net(x).clamp(-1,1)

            # ensure tensors are on the same device and clamped
            sr_lp  = sr_m11.clamp(-1,1).to(device)
            hr_lp  = hr_m11.to(device).clamp(-1,1)
            lr_lp  = lr.to(device).clamp(-1,1)

            # repeat channel to 3 for LPIPS
            sr_rgb = sr_lp.repeat(1, 3, 1, 1)
            hr_rgb = hr_lp.repeat(1, 3, 1, 1)
            lr_rgb = lr_lp.repeat(1, 3, 1, 1)

            with torch.no_grad():
                lpips_val_model = lpips_model(sr_rgb, hr_rgb).mean().item()
                lpips_val_base  = lpips_model(lr_rgb, hr_rgb).mean().item()

            # map to [0,1] for metrics
            sr01 = ((sr_m11 + 1)/2).cpu()
            hr01 = ((hr_m11 + 1)/2).cpu()
            lr01 = ((lr     + 1)/2).cpu()

            psnr_model = _psnr(sr01, hr01, mask)
            rmsd_model = _rmsd(sr01, hr01, mask)
            psnr_base  = _psnr(lr01, hr01, mask)
            rmsd_base  = _rmsd(lr01, hr01, mask)

            uid = batch.get('uid',['N/A'])[0]
            slc = batch.get('slice',[-1])[0]
            slc = slc.item() if torch.is_tensor(slc) else slc
            results.append({
                "uid": uid, "slice": slc,
                "psnr_model": psnr_model, "rmsd_model": rmsd_model,
                "psnr_baseline": psnr_base, "rmsd_baseline": rmsd_base,
                "lpips_model": lpips_val_model, "lpips_baseline": lpips_val_base
            })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    df.to_csv(args.outcsv, index=False)
    print("[saved]", args.outcsv)
    print(df.describe())
