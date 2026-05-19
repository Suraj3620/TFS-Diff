import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
    # Load the data
    df_attn = pd.read_csv('ablation_results_ATTENTION.csv')
    df_no_attn = pd.read_csv('ablation_results_NO_ATTENTION.csv')
    
    # Extract High-Dose RMSD arrays
    # We multiply by 1000 to make the numbers easier to read on a graph (e.g., 0.005 -> 5.0)
    err_attn = df_attn['Model_HighDose_RMSD'].values * 1000
    err_no_attn = df_no_attn['Model_HighDose_RMSD'].values * 1000

    # 1. CALCULATE MEAN ± STD
    mean_attn, std_attn = np.mean(err_attn), np.std(err_attn)
    mean_no_attn, std_no_attn = np.mean(err_no_attn), np.std(err_no_attn)

    print("="*60)
    print("STATISTICAL RELEVANCE REPORT (x10^3 for readability)")
    print("="*60)
    print(f"U-Net (Cross-Attn) High-Dose RMSD : {mean_attn:.3f} ± {std_attn:.3f}")
    print(f"U-Net (No Attn)    High-Dose RMSD : {mean_no_attn:.3f} ± {std_no_attn:.3f}")

    # 2. PERFORM T-TEST (Statistical Significance)
    t_stat, p_value = stats.ttest_rel(err_attn, err_no_attn)
    print(f"\nPaired T-Test Results:")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.4f}")
    
    if p_value > 0.05:
        print("\nCONCLUSION: p > 0.05. The difference is NOT statistically significant.")
        print("This PROVES the Professor's theory: Attention did not magically improve raw accuracy, it merely provided explainability!")
    else:
        print("\nCONCLUSION: p < 0.05. The difference IS statistically significant.")
    print("="*60)

    # 3. PLOT THE ERROR HISTOGRAM
    plt.figure(figsize=(10, 6))
    
    # Plot histograms with transparency
    plt.hist(err_no_attn, bins=20, alpha=0.5, label='U-Net (No Attn)', color='#4A90E2', edgecolor='black')
    plt.hist(err_attn, bins=20, alpha=0.5, label='U-Net (Cross-Attn)', color='#D0021B', edgecolor='black')
    
    # Add vertical lines for the 95th Percentile
    p95_attn = np.percentile(err_attn, 95)
    plt.axvline(p95_attn, color='#D0021B', linestyle='dashed', linewidth=2, label=f'95th %ile (Attn): {p95_attn:.2f}')

    plt.title('Error Histogram: High-Dose Beam RMSD Distribution (100 Samples)', fontsize=14, fontweight='bold')
    plt.xlabel('High-Dose RMSD (Scaled x10^3 for readability)', fontsize=12)
    plt.ylabel('Number of Patient Samples (Frequency)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('error_histogram.png', dpi=300)
    print("\n[*] Saved 'error_histogram.png'")

if __name__ == "__main__":
    main()