import os, sys
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

from tqdm import tqdm
import torch, numpy as np
import core.logger as Logger
import data as Data
import model as Model

def main():
    json_cfg = 'config/ct8_infer.json'
    
    args = type('Args', (), {
        'config': json_cfg,
        'phase': 'val',
        'gpu_ids': None,
        'debug': False,
        'enable_wandb': False,
        'log_infer': False
    })()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # ----------------------------------------------------------------
    val_set    = Data.create_dataset(opt['datasets']['val'], phase='val')
    val_loader = Data.create_dataloader(val_set,
                                        opt['datasets']['val'],
                                        phase='val')

    # First, create the model architecture from the config.
    # Then, load the pre-trained weights into it.
    diffusion = Model.create_model(opt)
    diffusion.load_network()

    # Set up the directory where results are saved.
    out_dir = 'inference_results'; os.makedirs(out_dir, exist_ok=True)
    diffusion.netG.eval()

    # `torch.no_grad()` is important for inference as it saves memory and speeds things up.
    with torch.no_grad():
        # Loop through each input file from the inference dataset.
        for idx, batch in enumerate(tqdm(val_loader)):
            diffusion.feed_data(batch)
            diffusion.test() # Run the main denoising process.
            
            visuals = diffusion.get_current_visuals()
            sr_3_chan = visuals['SR']
            
            # The model's raw output might have multiple channels, but only the first one is needed(the dose).
            sr_1_chan = sr_3_chan[:, 0, :, :]

            filename_stem = batch['filename_stem'][0]
            
            # Save the final result as a numpy array.
            np.save(f'{out_dir}/{filename_stem}.npy',
                    sr_1_chan.cpu().numpy())

if __name__ == '__main__':
    main()