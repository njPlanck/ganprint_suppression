import os
import cv2
import argparse
import numpy as np
from filters import nl_means_filter, bm3d_filter, sigma_filter, wavelet_filter
from prnu_cleaning import load_images_from_folder
from skimage import io, img_as_float, img_as_ubyte


def save_images(images, filenames, denoised_img, out_folder, filter_name, original_filename):
    """
    Saves the denoised image with proper conversion for visualization.
    """
    os.makedirs(out_folder, exist_ok=True)

    # Clip the values to a valid range and convert to 8-bit integer.
    denoised_img_clipped = np.clip(denoised_img, 0.0, 1.0)
    denoised_img_clipped = np.nan_to_num(denoised_img_clipped)
    denoised_img_8bit = (denoised_img_clipped * 255).astype(np.uint8)

    # Define save path using the original filename
    base_name = os.path.splitext(original_filename)[0]
    save_path = os.path.join(out_folder, f"{filter_name}_{base_name}_denoised.png")

    # Save using OpenCV
    cv2.imwrite(save_path, denoised_img_8bit)
    # print(f"[INFO] Saved denoised image: {save_path}")


def process_images_with_denoisers(root_folder, output_folder):
    """
    Applies denoising filters to images and saves the denoised results.
    """
    filters = {
        "nlm": nl_means_filter,
        "bm3d": bm3d_filter,
        "wavelet": wavelet_filter,
        "sigma": sigma_filter,
    }

    for gan_model in os.listdir(root_folder):
        gan_folder = os.path.join(root_folder, gan_model)
        if os.path.isdir(gan_folder):
            print(f"[INFO] Processing images from {gan_model} folder...")
            images, filenames = load_images_from_folder(gan_folder)
            
            if images:
                for filter_name, denoise_filter_func in filters.items():
                    print(f"[INFO] Applying {filter_name} filter...")
                    
                    # Create a dedicated output subfolder for each filter
                    filter_out_folder = os.path.join(output_folder, gan_model, filter_name)
                    
                    for img, fname in zip(images, filenames):
                        denoised_img = denoise_filter_func(img) 
                        
                        save_images(images, filenames, denoised_img, filter_out_folder, filter_name, fname)

                print(f"[INFO] All processing for {gan_model} complete.")
            else:
                print(f"[WARN] No images found in {gan_folder}. Skipping.")


# --- Main Script ---
# Define paths
root_folder = '/home/chinasa/python_projects/denoising/images/idiap/synthetic'
output_folder = '/home/chinasa/python_projects/denoising/output/denoised_idiap' # Changed output folder name for clarity

try:
    process_images_with_denoisers(root_folder, output_folder)
    print("[INFO] Script finished successfully.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")    