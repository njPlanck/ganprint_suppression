import os
from pathlib import Path
import cv2
import numpy as np


def compute_image_spectrum(image_path, target_shape=None, log_scale=True):
    """Loads image in grayscale, optionally resizes to match fingerprint, and computes FFT."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # FORCE SHAPE MATCHING: Resize the image if it doesn't match the mean spectrum
    if target_shape is not None and img.shape != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0

    # Compute 2D FFT & shift zero-frequency to center
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    spectrum = np.abs(f_shift)

    if log_scale:
        spectrum = np.log1p(spectrum)

    return spectrum


def create_adaptive_filter(mean_real_spectrum, synth_spectrum, num_stds=3, save_path="adaptive_filter_mask.png"):
    if mean_real_spectrum.shape != synth_spectrum.shape:
        raise ValueError(f"Shape mismatch: Mean {mean_real_spectrum.shape} vs Synth {synth_spectrum.shape}")

    h, w = mean_real_spectrum.shape

    difference_map = np.maximum(synth_spectrum - mean_real_spectrum, 0)

    diff_mean = np.mean(difference_map)
    diff_std = np.std(difference_map)
    adaptive_threshold = diff_mean + (num_stds * diff_std)

    # Form the filter (mask) where the difference exceeds the threshold
    filter_mask = (difference_map > adaptive_threshold).astype(np.float32)

    # Protect the critical central low-frequencies from being filtered
    center_y, center_x = h // 2, w // 2
    
    # Resolution scaling: dynamically scale the radius based on image dimensions
    base_dim = min(h, w)
    radius = max(15, int(base_dim * 0.025))  # Scales with size, minimum 15

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Force the core structure area to 0 (no filtering)
    filter_mask[dist_from_center <= radius] = 0

    # Apply morphological dilation (your original code)
    kernel = np.ones((3, 3), np.uint8)
    filter_mask = cv2.dilate(filter_mask, kernel, iterations=1)

    # Save and return
    mask_uint8 = (filter_mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8)

    return difference_map, filter_mask, adaptive_threshold


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
    attenuated_f_shift = f_shift * (1.0 - (1.5 * filter_mask))

    # 4. Perform Inverse FFT to reconstruct the image
    f_ishift = np.fft.ifftshift(attenuated_f_shift)
    img_back = np.fft.ifft2(f_ishift)
    
    # 5. Extract real spatial data and clip to valid 0-1 pixel range
    img_reconstructed = np.real(img_back)
    img_reconstructed = np.clip(img_reconstructed, 0.0, 1.0)

    return img, img_reconstructed


if __name__ == "__main__":
    # --- Paths ---
    MEAN_REAL_SPECTRUM_PATH = "/home/chinasa/python_projects/ganprint_removal/inverse_filtering/gan_noise/scut_fingerprint_spoofed.npy" 
    INPUT_GAN_FOLDER = Path("/home/chinasa/python_projects/pad_dataset/scut/synthetic/cycleGAN")
    OUTPUT_GAN_FOLDER = Path("/home/chinasa/python_projects/ganprint_removal/outputs/images/inverse_filtering/scut/cycleGAN_denoised")

    # Create the output directory if it doesn't exist
    OUTPUT_GAN_FOLDER.mkdir(parents=True, exist_ok=True)

    # 1. Load the pre-calculated Mean Real Spectrum once (outside the loop for efficiency)
    try:
        mean_real_spectrum = np.load(MEAN_REAL_SPECTRUM_PATH)
        if mean_real_spectrum.ndim > 2:
            mean_real_spectrum = np.squeeze(mean_real_spectrum)
    except FileNotFoundError:
        raise FileNotFoundError(f"Mean real spectrum not found at {MEAN_REAL_SPECTRUM_PATH}")

    # 2. Get all valid image files in the input folder
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_paths = [p for p in INPUT_GAN_FOLDER.iterdir() if p.suffix.lower() in valid_extensions]

    print(f"Found {len(image_paths)} images to process in {INPUT_GAN_FOLDER}")

    # 3. Loop through each image and process
    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        try:
            # Compute magnitude spectrum for the target synthetic image
            synth_spectrum = compute_image_spectrum(
                img_path,
                target_shape=mean_real_spectrum.shape
            )

            # Compute difference and generate the adaptive filter
            difference_map, filter_mask, threshold_val = create_adaptive_filter(
                mean_real_spectrum, 
                synth_spectrum, 
                num_stds=3,
                save_path="adaptive_filter_mask.png" 
            )

            # Apply the mask to the raw complex FFT and reconstruct the spatial image
            original_spatial_img, cleaned_spatial_img = reconstruct_cleaned_image(
                img_path, 
                filter_mask, 
                target_shape=mean_real_spectrum.shape
            )

            # 4. Save the denoised image to the new output folder
            output_file_path = OUTPUT_GAN_FOLDER / img_path.name
            cleaned_uint8 = (cleaned_spatial_img * 255.0).astype(np.uint8)
            cv2.imwrite(str(output_file_path), cleaned_uint8)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print(f"Batch processing complete! All denoised images saved to: {OUTPUT_GAN_FOLDER}")