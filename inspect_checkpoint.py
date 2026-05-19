import torch

def inspect_checkpoint(checkpoint_path):
    """
    Loads a PyTorch checkpoint and inspects the shape of a specific layer's weights.
    """
    try:
        # Load the checkpoint file.
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"Successfully loaded checkpoint: {checkpoint_path}\n")

        target_key = 'denoise_fn.downs.0.weight'

        if target_key in ckpt:
            weight_tensor = ckpt[target_key]
            print(f"Found the target layer: '{target_key}'")
            print(f"Shape of the weight tensor is: {weight_tensor.shape}")

            # The shape of a Conv2d weight is [out_channels, in_channels, kernel_height, kernel_width]
            in_channels = weight_tensor.shape[1]
            print(f"\nThis indicates the model's required 'in_channel' is: {in_channels}")
            print("\n==> Please update the 'in_channel' in your ct8_infer.json file to this value. <==")

        else:
            print(f"Error: The key '{target_key}' was NOT found in the checkpoint.")
            print("\nAvailable top-level keys in the checkpoint are:")
            for key in ckpt.keys():
                if 'denoise_fn' in key:
                    print(f"- {key}")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        print("Please ensure the path is correct and you are running this from the project root.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # The path to specific checkpoint file.
    path_to_checkpoint = 'checkpoint/I490000_E61250_gen.pth'
    inspect_checkpoint(path_to_checkpoint)