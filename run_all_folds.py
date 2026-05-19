import json
import os

def run_kfold(base_config_path, model_prefix):
    print(f"\n{'='*50}")
    print(f"STARTING K-FOLD CROSS VALIDATION FOR: {model_prefix}")
    print(f"{'='*50}")

    for fold in range(1, 6):
        # 1. Load the base config
        with open(base_config_path, 'r') as f:
            opt = json.load(f)

        # 2. Inject the correct fold paths
        opt['datasets']['train']['dataroot'] = f"data/CT_Dose_Dataset/folds/fold_{fold}/train_index.csv"
        opt['datasets']['val']['dataroot'] = f"data/CT_Dose_Dataset/folds/fold_{fold}/val_index.csv"
        
        # 3. Give it a unique save name (e.g., CV_Attn_Fold_1)
        opt['name'] = f"CV_{model_prefix}_Fold_{fold}"
        
        # 4. Save to a temporary config file
        temp_config = f"config/temp_{model_prefix}_fold_{fold}.json"
        with open(temp_config, 'w') as f:
            json.dump(opt, f, indent=4)

        # 5. Run the training script
        print(f"\n[*] Launching {model_prefix} - FOLD {fold}/5...")
        # Use os.system to run the exact same command you would type in the terminal
        exit_code = os.system(f"python3 sr.py -c {temp_config} -p train")
        
        if exit_code != 0:
            print(f"[!] Warning: Fold {fold} did not exit cleanly. Stopping automation.")
            break

        print(f"[*] Finished {model_prefix} - FOLD {fold}/5")

if __name__ == "__main__":
    # Ensure you are running on the correct GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 1. Run all 5 folds for the Attention Model
    # run_kfold("config/cv_attn_base.json", "Attn")
    
    # 2. Run all 5 folds for the No-Attention Model
    # (Uncomment this when you are ready to run the second model, or let it run both!)
    run_kfold("config/cv_no_attn_base.json", "No_Attn")
    
    print("\n[SUCCESS] ALL K-FOLD RUNS COMPLETED!")