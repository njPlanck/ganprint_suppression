import os
import glob
import numpy as np
from PIL import Image

# 1. Standard LBP Function
def standard_lbp_histogram(image):
    """
    Computes the Standard LBP (8 neighbors, radius 1) 
    and returns a simple normalized frequency histogram.
    """
    img_float = np.asarray(image, dtype=np.float32)
    
    # Pad image to handle borders
    padded = np.pad(img_float, pad_width=1, mode='edge')
    
    # Extract 3x3 patch neighbors (clockwise or counter-clockwise)
    neighbors = np.stack([
        padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:], # Top row
        padded[1:-1, 2:],                                         # Right
        padded[2:, 2:],     padded[2:, 1:-1], padded[2:, 0:-2], # Bottom row
        padded[1:-1, 0:-2]                                       # Left
    ], axis=-1)
    
    center = img_float[..., np.newaxis]
    
    # Compare neighbors DIRECTLY to the center pixel
    binary_codes = (neighbors >= center).astype(np.uint8)
    
    # Standard powers of two for an 8-bit integer (0-255)
    powers_of_two = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    lbp_image = np.sum(binary_codes * powers_of_two, axis=-1).astype(np.uint8)
    
    # Create Frequency Histogram (Unweighted)
    hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
    
    # Normalize
    hist = hist.astype("float32")
    if np.sum(hist) > 0:
        hist /= np.sum(hist)
        
    return hist

# 2. Process Files directly from glob list
def get_features_from_file_list(file_list, category_name):
    features = []
    print(f"Extracting features from {len(file_list)} {category_name} images...")
    
    for path in file_list:
        try:
            img = Image.open(path).convert('L')
            features.append(standard_lbp_histogram(img))
        except Exception as e:
            print(f"  [Error] Skipping {path}: {e}")
            
    return np.array(features)

# 3. Combine and Export
def generate_dataset_from_globs(real_files, fake_files, synthetic_files, output_csv_path):
    # Step 1: Extract features from the lists
    gen_features = get_features_from_file_list(real_files, "Real")
    spoof_features = get_features_from_file_list(fake_files, "Spoofed")
    gan_features = get_features_from_file_list(synthetic_files, "Synthetic")
    
    # Validation check to ensure paths actually found files
    if len(gen_features) == 0 or len(spoof_features) == 0 or len(gan_features) == 0:
        print("\n[Error] One or more datasets returned 0 files. Please check your glob paths.")
        return

    # Step 2: Combine and Label
    X = np.vstack((gen_features, spoof_features, gan_features))
    y = np.concatenate((np.zeros(len(gen_features)), 
                        np.ones(len(spoof_features)), 
                        np.full(len(gan_features), 2)))

    # Step 3: Save to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    final_table = np.column_stack((y, X))
    header = "label," + ",".join([f"bin_{i}" for i in range(256)])
    np.savetxt(output_csv_path, final_table, delimiter=",", header=header, comments="", fmt="%g")
    
    print(f"\nSuccessfully saved dataset to: {output_csv_path}")

if __name__ == "__main__":
    # Define your exact file paths using glob
    real_folder = glob.glob(r'/home/chinasa/python_projects/pad_dataset/scut/genuine/**/*.bmp', recursive=True)
    fake_folder = glob.glob(r'/home/chinasa/python_projects/pad_dataset/scut/spoofed/**/*.bmp', recursive=True)
    synthetic_folder = glob.glob(r'/home/chinasa/python_projects/pad_dataset/scut/synthetic/dritGAN/*.png', recursive=True) 
    
    # Define where the CSV should be saved
    output_csv = "outputs/csvs/scut/lbp/lbp_noisy_scut_dritGAN.csv"
    
    # Run the generator
    generate_dataset_from_globs(real_folder, fake_folder, synthetic_folder, output_csv)