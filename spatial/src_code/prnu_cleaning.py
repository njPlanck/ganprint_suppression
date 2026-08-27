import os
import cv2
import argparse
import numpy as np
from filters import nl_means_filter, bm3d_filter, sigma_filter, wavelet_filter, extract_fingerprint, fingerprint_to_uint8
from skimage import io, img_as_float, img_as_ubyte



def average_fingerprint(images, denoise_filter):
    """
    Computes the average fingerprint from a set of images.
    The fingerprint of a single image is defined as: Image - Denoised_Image.
    This function handles inconsistencies in image dimensions, which is a common
    issue with wavelet-based denoising.
    """
    if not images:
        return None

    # Determine a common shape to resize all images to.
    # We'll use the shape of the first image as the reference.
    ref_shape = images[0].shape
    
    # Initialize the average fingerprint with zeros based on the reference shape
    fingerprint_average = np.zeros(ref_shape)

    # Process and resize each image
    valid_images = 0
    for img in images:
        # Check if the image has a consistent shape; resize if needed.
        if img.shape != ref_shape:
            img = cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Extract and add the fingerprint
        fingerprint = extract_fingerprint(img, denoise_filter)
        
        # Handle cases where the fingerprint might have slightly different dimensions
        # after filtering, by resizing it to the reference shape.
        if fingerprint.shape != ref_shape:
            fingerprint = cv2.resize(fingerprint, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR)
            
        fingerprint_average += fingerprint
        valid_images += 1

    if valid_images == 0:
        # Return a zero array if no valid images were processed
        return np.zeros(ref_shape)
        
    return fingerprint_average / valid_images


def load_images_from_folder(folder_path):
    """
    Loads all supported image files from a given folder, converts them to float,
    and returns them along with their filenames.
    """
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"[WARN] Failed to read: {img_path}")
                continue
            # Convert to float and normalize to [0, 1] for consistent processing
            image = image.astype(np.float32) / 255.0
            images.append(image)
            filenames.append(filename)
    return images, filenames


def save_fingerprint_removed_images(images, filenames, avg_fingerprint, out_folder, filter_name, strength=1):
    """
    Subtracts the average fingerprint from each image and saves the resulting
    "clean" images with proper conversion for visualization.
    """
    os.makedirs(out_folder, exist_ok=True)

    for img, fname in zip(images, filenames):
        # 1. Subtract the average fingerprint to get the clean image
        clean_img = img - (strength * avg_fingerprint)
        
        # 2. Clip the values to a valid range to prevent out-of-bounds issues
        # and then convert to 8-bit integer.
        # This is the crucial step. It assumes the result is still a full image.
        clean_img_clipped = np.clip(clean_img, 0.0, 1.0)
        clean_img_clipped = np.nan_to_num(clean_img_clipped)
        # 3. Convert to 8-bit unsigned integer
        clean_img_8bit = (clean_img_clipped * 255).astype(np.uint8)
        
        # 4. Define save path
        base_name = os.path.splitext(fname)[0]
        save_path = os.path.join(out_folder, f"{filter_name}_{base_name}_clean.png")

        # 5. Save using OpenCV
        cv2.imwrite(save_path, clean_img_8bit)

    print(f"[INFO] Saved {len(images)} clean images to {out_folder}")


def build_and_apply_fingerprints(root_folder, output_folder):
    """
    Iterates through GAN models, computes average fingerprints, and saves
    the resulting fingerprint-removed images.
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
                for filter_name, denoise_filter in filters.items():
                    print(f"[INFO] Computing average fingerprint with {filter_name} filter...")
                    avg_fingerprint = average_fingerprint(images, denoise_filter)

                    # Output folder: output/gan_model/filter_name/
                    out_folder = os.path.join(output_folder, gan_model, filter_name)
                    save_fingerprint_removed_images(images, filenames, avg_fingerprint, out_folder, filter_name)

                print(f"[INFO] All processing for {gan_model} complete.")
            else:
                print(f"[WARN] No images found in {gan_folder}. Skipping.")


# --- Corrected Main Script ---
# Define paths

if __name__ == "__main__":
    root_folder = '/home/chinasa/python_projects/denoising/images/plus/synthetic'
    output_folder = '/home/chinasa/python_projects/denoising/output/cleaned_plus'
    try:
        # This is the corrected part.
        # Call the build_and_apply_fingerprints function directly
        # with the appropriate input and output folders.
        build_and_apply_fingerprints(root_folder, output_folder)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")