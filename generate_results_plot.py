import matplotlib.pyplot as plt
import numpy as np

def main():
    # --- The Data ---
    models =['Baseline\n(B-Spline)', 'U-Net\n(No Attn)', 'U-Net\n(Cross-Attn)', 'DiT\n(Transformer)']
    
    global_rmsd =[0.00478, 0.00158, 0.00172, 0.00208]
    high_dose_rmsd =[0.04191, 0.00574, 0.00572, 0.00660]
    
    # Inference speeds in minutes (Baseline is instantaneous, so we leave it out of the speed plot or set to 0)
    models_ai =['U-Net\n(No Attn)', 'U-Net\n(Cross-Attn)', 'DiT\n(Transformer)']
    speeds_mins =[27.0, 32.0, 2.86] # 2m52s = 2.86 mins

    # --- Setup Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Ablation Study: Model Performance Comparison', fontsize=18, fontweight='bold', y=1.05)

    colors =['#8C8C8C', '#4A90E2', '#D0021B', '#F5A623']
    
    # 1. High-Dose RMSD (The most critical metric)
    axes[0].bar(models, high_dose_rmsd, color=colors, edgecolor='black')
    axes[0].set_title('High-Dose Beam Error (RMSD)\nLower is better', fontsize=14)
    axes[0].set_ylabel('RMSD')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    # Add data labels
    for i, v in enumerate(high_dose_rmsd):
        axes[0].text(i, v + 0.001, f"{v:.5f}", ha='center', fontweight='bold')

    # 2. Global RMSD
    axes[1].bar(models, global_rmsd, color=colors, edgecolor='black')
    axes[1].set_title('Global Body Error (RMSD)\nLower is better', fontsize=14)
    axes[1].set_ylabel('RMSD')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    # Add data labels
    for i, v in enumerate(global_rmsd):
        axes[1].text(i, v + 0.0001, f"{v:.5f}", ha='center', fontweight='bold')

    # 3. Inference Speed (AI Models Only)
    colors_speed =['#4A90E2', '#D0021B', '#F5A623']
    axes[2].bar(models_ai, speeds_mins, color=colors_speed, edgecolor='black')
    axes[2].set_title('Inference Speed (100 Samples)\nLower is faster', fontsize=14)
    axes[2].set_ylabel('Time (Minutes)')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    # Add data labels
    for i, v in enumerate(speeds_mins):
        axes[2].text(i, v + 0.5, f"{v:.1f} min", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_ablation_summary.png', dpi=300, bbox_inches='tight')
    print("[*] Saved visualization to 'final_ablation_summary.png'")

if __name__ == "__main__":
    main()