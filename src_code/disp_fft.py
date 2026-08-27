import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def compute_fft(image_path):
    """Loads an image, converts to grayscale, and computes its FFT magnitude spectrum."""
    if not os.path.exists(image_path):
        print(f"Error: Could not find {image_path}")
        return None, None

    # Load image and ensure it is grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Compute the 2D Fast Fourier Transform
    f_transform = np.fft.fft2(img_array)
    
    # Shift the zero-frequency component to the center of the spectrum
    f_shift = np.fft.fftshift(f_transform)
    
    # Calculate magnitude spectrum (use log scale because FFT values vary massively)
    # Adding 1 ensures we don't calculate log(0)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    return img_array, magnitude_spectrum

def plot_fft_comparison(real_path, synthetic_path, denoised_path):
    # Process all three images
    real_img, real_fft = compute_fft(real_path)
    synth_img, synth_fft = compute_fft(synthetic_path)
    denoised_img, denoised_fft = compute_fft(denoised_path)

    if any(x is None for x in [real_img, synth_img, denoised_img]):
        print("Exiting due to missing files.")
        return

    # Set up the plot (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.canvas.manager.set_window_title('Fingerprint FFT Analysis')

    # Row 1: Original Images
    axes[0, 0].imshow(real_img, cmap='gray')
    axes[0, 0].set_title('Real Fingerprint')
    
    axes[0, 1].imshow(synth_img, cmap='gray')
    axes[0, 1].set_title('Synthetic (Noisy)')
    
    axes[0, 2].imshow(denoised_img, cmap='gray')
    axes[0, 2].set_title('Denoised (Autoencoder)')

    # Row 2: FFT Magnitude Spectra
    axes[1, 0].imshow(real_fft, cmap='gray')
    axes[1, 0].set_title('Real FFT')
    
    axes[1, 1].imshow(synth_fft, cmap='gray')
    axes[1, 1].set_title('Synthetic FFT')
    
    axes[1, 2].imshow(denoised_fft, cmap='gray')
    axes[1, 2].set_title('Denoised FFT')

    # Clean up axes
    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    
    # Save the plot to your outputs folder
    save_path = "fft_comparison_denoised_scut.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to {save_path}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS TO YOUR ACTUAL IMAGES ---
    # You can crop a single real, synthetic, and denoised image from your epoch_150.png 
    # or point directly to individual files in your dataset/output folders.
    
   
    PATH_REAL = "/home/chinasa/python_projects/pad_dataset/scut/spoofed/001/1_1_0_2_3.bmp"
    PATH_SYNTHETIC = "/home/chinasa/python_projects/pad_dataset/scut/synthetic/cycleGAN/001-1_1_1_1_5.png"
    PATH_DENOISED = "/home/chinasa/python_projects/auto_encoder/outputs/images/scut_fft/cycleGAN/001-1_1_1_1_5.png"
    
    plot_fft_comparison(PATH_REAL, PATH_SYNTHETIC, PATH_DENOISED)