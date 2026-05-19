import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # File paths
    files = {
        'U-Net\n(No Attn)': 'ablation_results_NO_ATTENTION.csv',
        'U-Net\n(Cross-Attn)': 'ablation_results_ATTENTION.csv',
        'DiT\n(Transformer)': 'ablation_results_DIT.csv'
    }

    results =[]

    for name, path in files.items():
        try:
            df = pd.read_csv(path)
            
            # 1. Standard Error Metrics
            global_rmsd = df['Model_Global_RMSD'].mean()
            high_dose_rmsd = df['Model_HighDose_RMSD'].mean()
            
            # 2. Clinical Safety Metrics (Professor's Request)
            # Filter out 0.0 values (slices where no high-dose beam was present)
            valid_95th = df[df['Model_HighDose_95th_Error'] > 0.0]['Model_HighDose_95th_Error']
            valid_max = df[df['Model_HighDose_MAX_Error'] > 0.0]['Model_HighDose_MAX_Error']
            
            avg_95th_err = valid_95th.mean()
            avg_max_err = valid_max.mean()
            
            results.append({
                'Model': name,
                'Global RMSD': global_rmsd,
                'High-Dose RMSD': high_dose_rmsd,
                '95th Percentile Error': avg_95th_err,
                'Max Error (Worst-Case)': avg_max_err
            })
            
        except FileNotFoundError:
            print(f"[!] Could not find {path}. Skipping...")

    # --- Print Terminal Table ---
    results_df = pd.DataFrame(results)
    
    # We will also add the Baseline (B-Spline) metrics extracted from the U-Net CSV
    baseline_df = pd.read_csv('ablation_results_ATTENTION.csv')
    base_global = baseline_df['Base_Global_RMSD'].mean()
    base_high = baseline_df['Base_HighDose_RMSD'].mean()
    
    print("\n" + "="*90)
    print("FINAL CLINICAL SAFETY & ABLATION TABLE")
    print("="*90)
    print(f"{'Model':<22} | {'Global RMSD':<13} | {'High-Dose RMSD':<16} | {'95th %ile Error':<17} | {'Max Error':<10}")
    print("-" * 90)
    print(f"{'Baseline (B-Spline)':<22} | {base_global:<13.5f} | {base_high:<16.5f} | {'N/A':<17} | {'N/A':<10}")
    for _, row in results_df.iterrows():
        print(f"{row['Model'].replace(chr(10), ' '):<22} | {row['Global RMSD']:<13.5f} | {row['High-Dose RMSD']:<16.5f} | {row['95th Percentile Error']:<17.5f} | {row['Max Error (Worst-Case)']:<10.5f}")
    print("="*90)

    # --- Generate the Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Clinical Safety & Ablation Study Comparison', fontsize=18, fontweight='bold', y=1.05)

    models = results_df['Model'].tolist()
    colors =['#4A90E2', '#D0021B', '#F5A623']

    # 1. High-Dose RMSD
    axes[0].bar(models, results_df['High-Dose RMSD'], color=colors, edgecolor='black')
    axes[0].set_title('High-Dose Beam Error (RMSD)\nLower is better', fontsize=14)
    axes[0].set_ylabel('RMSD')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(results_df['High-Dose RMSD']):
        axes[0].text(i, v + 0.0001, f"{v:.5f}", ha='center', fontweight='bold')

    # 2. 95th Percentile Error
    axes[1].bar(models, results_df['95th Percentile Error'], color=colors, edgecolor='black')
    axes[1].set_title('95th Percentile Error (Clinical Safety)\nLower is better', fontsize=14)
    axes[1].set_ylabel('Absolute Error')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(results_df['95th Percentile Error']):
        axes[1].text(i, v + 0.0005, f"{v:.5f}", ha='center', fontweight='bold')

    # 3. Maximum Error (Worst Case)
    axes[2].bar(models, results_df['Max Error (Worst-Case)'], color=colors, edgecolor='black')
    axes[2].set_title('Average Maximum Error (Worst-Case)\nLower is better', fontsize=14)
    axes[2].set_ylabel('Absolute Error')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(results_df['Max Error (Worst-Case)']):
        axes[2].text(i, v + 0.001, f"{v:.5f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('clinical_safety_summary.png', dpi=300, bbox_inches='tight')
    print("\n[*] Saved clinical safety visualization to 'clinical_safety_summary.png'")

if __name__ == "__main__":
    main()