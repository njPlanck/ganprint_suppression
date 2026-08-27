import os
import glob
import numpy as np
import cv2
from scipy.fftpack import dctn, idctn
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def compute_dct(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    return dctn(np.float32(img), type=2, norm='ortho')


#def compute_fingerprint_lasso(genuine_paths, synthetic_paths, alpha=0.00000001):
def compute_fingerprint_lasso(genuine_paths, synthetic_paths, alpha=0.00000001): #reduced the alpha by 10
    # Step 1: Extract and flatten DCT features
    X = []
    y = []

    for path in tqdm(genuine_paths, desc="Genuine DCTs"):
        dct_img = compute_dct(path)
        X.append(dct_img.flatten())
        y.append(0)  # Label: genuine

    for path in tqdm(synthetic_paths, desc="Synthetic DCTs"):
        dct_img = compute_dct(path)
        X.append(dct_img.flatten())
        y.append(1)  # Label: synthetic

    X = np.array(X)
    y = np.array(y)

    # Step 2: Normalize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Train Lasso regression model
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    # Step 4: Get regression weights and reshape to original DCT shape
    weights = lasso.coef_
    example_dct = compute_dct(genuine_paths[0])
    dct_shape = example_dct.shape
    fingerprint = weights.reshape(dct_shape)
    return fingerprint


def inverse_dct(dct_coeffs):
    return idctn(dct_coeffs, type=2, norm='ortho')

def scale_fp_to_neg1_1(fp):
    min_val = np.min(fp)
    max_val = np.max(fp)
    return 2 * (fp - min_val) / (max_val - min_val + 1e-8) - 1

def apply_lasso_spectrum_attack(image_path, fingerprint, strength=10):
    original_dct = compute_dct(image_path)
    fingerprint = scale_fp_to_neg1_1(strength * fingerprint)
    attacked_dct = original_dct * (1 -  fingerprint)
    attacked_img = inverse_dct(attacked_dct)
    attacked_img = np.clip(attacked_img, 0, 255).astype(np.uint8)
    return attacked_img

def process_and_save_attacks(input_dir, output_dir, fingerprint):
    image_paths = glob.glob(os.path.join(input_dir, '**/*.png'), recursive=True)

    for path in tqdm(image_paths, desc=f"Attacking {os.path.basename(input_dir)}"):
        rel_path = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            attacked_img = apply_lasso_spectrum_attack(path, fingerprint)
            cv2.imwrite(out_path, attacked_img)
        except Exception as e:
            print(f"Error processing {path}: {e}")

# === Main Script ===

genuine_paths = glob.glob(r'images/genuine/idiap/**/*.png', recursive=True)
#genuine_paths = glob.glob(r'images/genuine/scut/**/*.bmp', recursive=True)
#genuine_paths = glob.glob(r'images/genuine/idiap/*.png', recursive=True)
synthetic_root = r'images/synthetic/idiap'
output_root = r'lasso_output_10/synthetic_attacked/idiap'
synthetic_methods = ['cycleGAN', 'distanceGAN', 'dritGAN','starGAN']  # Add others here

for method in synthetic_methods:
    print(f"\n=== Processing method: {method} ===")

    synthetic_dir = os.path.join(synthetic_root, method)
    output_dir = os.path.join(output_root, method)
    synthetic_paths = glob.glob(os.path.join(synthetic_dir, '**/*.png'), recursive=True)

    if not synthetic_paths:
        print(f"No images found for method: {method}")
        continue

    fingerprint = compute_fingerprint_lasso(genuine_paths, synthetic_paths)
    process_and_save_attacks(synthetic_dir, output_dir, fingerprint)
