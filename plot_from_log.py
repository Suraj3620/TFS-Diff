import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_log_file(log_path):
    """Parses the training log file to extract validation metrics."""
    
    # Regex to find the iteration number from any training line
    iter_regex = re.compile(r".*iter:\s*([\d,]+)>")
    
    # Regex to specifically find lines with validation results
    validation_regex = re.compile(
        r"# Validation # PSNR: ([\d.]+), RMSD: ([\d.e-]+), LPIPS: ([\d.]+), Baseline RMSD: ([\d.e-]+)"
    )

    data = []
    last_iteration = 0
    
    print(f"[*] Reading log file: {log_path}")
    with open(log_path, 'r') as f:
        for line in f:
            # First, check if the line contains an iteration number and update it
            iter_match = iter_regex.search(line)
            if iter_match:
                # Remove commas from the iteration string and convert to int
                last_iteration = int(iter_match.group(1).replace(',', ''))

            # Next, check if the line is a validation line
            val_match = validation_regex.search(line)
            if val_match:
                try:
                    # Associate the validation metrics with the last seen iteration number
                    psnr = float(val_match.group(1))
                    rmsd = float(val_match.group(2))
                    lpips = float(val_match.group(3))
                    rmsd_baseline = float(val_match.group(4))
                    
                    data.append({
                        'iteration': last_iteration,
                        'psnr_mask': psnr,
                        'rmsd_mask': rmsd,
                        'lpips_mask': lpips,
                        'rmsd_base_mask_lr_up': rmsd_baseline
                    })
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse validation line: {line.strip()} -> {e}")

    if not data:
        print("[ERROR] No validation data found in the log file. Check the log file path and content.")
        return None
        
    print(f"[*] Found {len(data)} validation steps in the log.")
    return pd.DataFrame(data)

def plot_metrics(df, save_dir):
    """Creates and saves plots for all validation metrics."""
    
    if df is None or df.empty:
        print("[INFO] DataFrame is empty. No plots will be generated.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = {
        'psnr_mask': 'PSNR (Higher is Better)',
        'rmsd_mask': 'RMSD (Lower is Better)',
        'lpips_mask': 'LPIPS (Lower is Better)',
    }
    
    for metric_key, y_label in metrics_to_plot.items():
        plt.figure(figsize=(12, 7))
        
        sns.lineplot(data=df, x='iteration', y=metric_key, marker='o', label='Model Performance')
        
        if metric_key == 'rmsd_mask':
            # Add the baseline RMSD to the RMSD plot for direct comparison
            sns.lineplot(data=df, x='iteration', y='rmsd_base_mask_lr_up', marker='x', linestyle='--', label='Baseline (LR-up)')

        plt.title(f'Validation {y_label} vs. Training Iteration')
        plt.xlabel('Training Iteration')
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'validation_curve_{metric_key}.png')
        plt.savefig(save_path)
        print(f"[*] Saved plot to {save_path}")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse a training log and plot validation curves.")
    parser.add_argument(
        '--log_file', 
        type=str, 
        # Verify this default path is correct for the latest run
        default='/home/sdesai/THESIS/TFS-Diff-main/experiments/CT_Dose_SuperResolution_FinalRun_250925_195521/experiments/CT_Dose_SuperResolution_FinalRun/logs/train.log', 
        help='Path to the training log file.'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='experiments/validation_plots_final', 
        help='Directory to save the output plots.'
    )
    
    args = parser.parse_args()
    
    log_df = parse_log_file(args.log_file)
    plot_metrics(log_df, args.save_dir)