import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

def load_and_process_image(path, target_hw):
    raw_arr = np.load(path).astype(np.float32)
    if raw_arr.ndim == 3:
        raw_arr = raw_arr[0, :, :]
    resized_arr = resize(raw_arr, target_hw, order=3, mode='edge', anti_aliasing=False, preserve_range=True)
    clipped_arr = np.clip(resized_arr, -1000, 3000)
    normalized_arr = ((clipped_arr + 1000) / 2000.0) - 1.0
    return normalized_arr

def compare_results(image_filename, data_root="data/input", results_dir="inference_results"):
    target_hw = (128, 112)
    filename_stem = os.path.splitext(image_filename)[0]

    try:
        sr_path = os.path.join(results_dir, image_filename)
        sr_image = np.load(sr_path)[0, :, :]

        hr_path = os.path.join(data_root, 'hr_128', image_filename)
        hr_image = load_and_process_image(hr_path, target_hw)
        
        lr_path = os.path.join(data_root, 'lr_16_128', image_filename)
        lr_image_display = resize(hr_image, (16, 14), order=0) # Pixelated version for display

        data_range = 2.0
        psnr = peak_signal_noise_ratio(hr_image, sr_image, data_range=data_range)
        ssim = structural_similarity(hr_image, sr_image, data_range=data_range)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(lr_image_display, cmap='gray', vmin=-1, vmax=1)
        axes[0].set_title(f"Low-Res Input")
        axes[0].axis('off')
        axes[1].imshow(sr_image, cmap='gray', vmin=-1, vmax=1)
        axes[1].set_title("Generated SR (Output)")
        axes[1].axis('off')
        axes[2].imshow(hr_image, cmap='gray', vmin=-1, vmax=1)
        axes[2].set_title("High-Res (Ground Truth)")
        axes[2].axis('off')
        fig.suptitle(f"Comparison for {filename_stem}\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}", fontsize=16)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred in compare_results: {e}")

if __name__ == "__main__":
    results_dir = "inference_results"
    try:
        image_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.npy')])
        if image_files:
            compare_results(image_files[0])
        else:
            print(f"No files found in '{results_dir}'.")
    except FileNotFoundError:
        print(f"Error: The directory '{results_dir}' was not found.")