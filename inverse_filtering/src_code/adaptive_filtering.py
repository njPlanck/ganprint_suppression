import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_image_spectrum(image_path, target_shape=None, log_scale=True):
    """Loads image in grayscale, optionally resizes to match fingerprint, and computes FFT."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # FORCE SHAPE MATCHING: Resize the image if it doesn't match the mean spectrum
    if target_shape is not None and img.shape != target_shape:
        print(f"Resizing test image from {img.shape} to {target_shape} to match mean spectrum.")
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0

    # Compute 2D FFT & shift zero-frequency to center
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    spectrum = np.abs(f_shift)

    if log_scale:
        spectrum = np.log1p(spectrum)

    return spectrum


def create_adaptive_filter(mean_real_spectrum, synth_spectrum, num_stds=2.0):
    """
    Subtracts synthetic spectrum from the mean real spectrum to compute the difference,
    then applies adaptive thresholding to form a binary filter mask.
    """
    if mean_real_spectrum.shape != synth_spectrum.shape:
        raise ValueError(f"Shape mismatch: Mean {mean_real_spectrum.shape} vs Synth {synth_spectrum.shape}")

    # 1. Subtract synthetic FFT from the mean real spectrum
    difference_map = np.maximum(synth_spectrum - mean_real_spectrum, 0)

    # 2. Compute adaptive threshold based on the difference map
    diff_mean = np.mean(difference_map)
    diff_std = np.std(difference_map)
    adaptive_threshold = diff_mean + (num_stds * diff_std)

    # 3. Form the filter (mask) where the difference exceeds the threshold
    filter_mask = (difference_map > adaptive_threshold).astype(np.float32)

    return difference_map, filter_mask, adaptive_threshold


def apply_filter_to_synthetic(synth_spectrum, filter_mask):
    """Applies the generated adaptive filter to the synthetic magnitude spectrum."""
    filtered_spectrum = synth_spectrum * (1 - (1.3 * filter_mask))
    return filtered_spectrum


def reconstruct_cleaned_image(original_img_path, filter_mask, target_shape=None):
    """Applies the mask to the raw complex FFT data and reconstructs the spatial image using iFFT."""
    # 1. Load and prepare original image
    img = cv2.imread(str(original_img_path), cv2.IMREAD_GRAYSCALE)
    if target_shape is not None and img.shape != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    # 2. Compute the raw, complex FFT (preserving phase data, no log scale)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)

    # 3. Apply the attenuator mask directly to the complex data (matching your 1.02 scaling)
    attenuated_f_shift = f_shift * (1.0 - (1.02 * filter_mask))

    # 4. Perform Inverse FFT to reconstruct the image
    f_ishift = np.fft.ifftshift(attenuated_f_shift)
    img_back = np.fft.ifft2(f_ishift)
    
    # 5. Extract real spatial data and clip to valid 0-1 pixel range
    img_reconstructed = np.real(img_back)
    img_reconstructed = np.clip(img_reconstructed, 0.0, 1.0)

    return img, img_reconstructed


def plot_pipeline(mean_real, synth_spec, diff_map, filter_mask, filtered_synth, original_img, cleaned_img, output_path="../gan_noise/adaptive_filter.png"):
    """Visualizes the difference-based filtering pipeline alongside the spatial images."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # --- SPECTRAL PLOTS (Columns 1-3) ---
    axes[0, 0].imshow(mean_real, cmap="gray")
    axes[0, 0].set_title("1. Mean Real Spectrum")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(synth_spec, cmap="gray")
    axes[0, 1].set_title("2. Synthetic Image Spectrum")
    axes[0, 1].axis("off")

    im_diff = axes[0, 2].imshow(diff_map, cmap="gray")
    axes[0, 2].set_title("3. Difference: Max(Synth - Mean, 0)")
    axes[0, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(filter_mask, cmap="gray")
    axes[1, 0].set_title("4. Adaptive Filter Mask")
    axes[1, 0].axis("off")

    im_filt = axes[1, 1].imshow(filtered_synth, cmap="gray")
    axes[1, 1].set_title("5. Filtered Synthetic Spectrum")
    axes[1, 1].axis("off")
    fig.colorbar(im_filt, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].axis("off") # Empty space for layout

    # --- SPATIAL IMAGE PLOTS (Column 4) ---
    axes[0, 3].imshow(original_img, cmap="gray")
    axes[0, 3].set_title("6. Original Synthetic Image")
    axes[0, 3].axis("off")

    axes[1, 3].imshow(cleaned_img, cmap="gray")
    axes[1, 3].set_title("7. Cleaned Image (GAN-print Removed)")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # --- Paths ---
    MEAN_REAL_SPECTRUM_PATH = "/home/chinasa/python_projects/ganprint_removal/inverse_filtering/gan_noise/scut_fingerprint_spoofed.npy" 
    SYNTHETIC_IMAGE_PATH = "/home/chinasa/python_projects/pad_dataset/idiap/synthetic/cycleGAN/001-001_L_1.png"

    # 1. Load the pre-calculated Mean Real Spectrum
    try:
        mean_real_spectrum = np.load(MEAN_REAL_SPECTRUM_PATH)
        # Safety check: if the array saved with an extra color channel, remove it
        if mean_real_spectrum.ndim > 2:
            mean_real_spectrum = np.squeeze(mean_real_spectrum)
    except FileNotFoundError:
        print(f"Warning: {MEAN_REAL_SPECTRUM_PATH} not found. Generating dummy array.")
        mean_real_spectrum = np.random.rand(256, 256)

    # 2. Compute magnitude spectrum for the target synthetic image
    synth_spectrum = compute_image_spectrum(
        SYNTHETIC_IMAGE_PATH,
        target_shape=mean_real_spectrum.shape
    )

    # 3. Compute difference and generate the adaptive filter
    difference_map, filter_mask, threshold_val = create_adaptive_filter(
        mean_real_spectrum, 
        synth_spectrum, 
        num_stds=2.5  
    )
    print(f"Calculated Adaptive Threshold: {threshold_val:.4f}")

    # 4. Apply the filter to the synthetic image's magnitude spectrum (for visualization)
    filtered_synthetic = apply_filter_to_synthetic(synth_spectrum, filter_mask)

    # 5. Apply the mask to the raw complex FFT and reconstruct the spatial image
    original_spatial_img, cleaned_spatial_img = reconstruct_cleaned_image(
        SYNTHETIC_IMAGE_PATH, 
        filter_mask, 
        target_shape=mean_real_spectrum.shape
    )

    anomaly_score = np.sum(filtered_synthetic)
    print(f"Total isolated anomaly energy: {anomaly_score:.2f}")

    # 6. Visualize the exact steps of your pipeline including the spatial images
    plot_pipeline(
        mean_real_spectrum, 
        synth_spectrum, 
        difference_map, 
        filter_mask, 
        filtered_synthetic,
        original_spatial_img,
        cleaned_spatial_img
    )