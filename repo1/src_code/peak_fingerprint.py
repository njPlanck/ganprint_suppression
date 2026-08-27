import os
import glob
import numpy as np
import cv2
from scipy.fftpack import dctn, idctn
from tqdm import tqdm

def compute_dct(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    return dctn(np.float32(img), type=2, norm='ortho')

def compute_peak_dct(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    eps = 1e-13
    dct_img = dctn(np.float32(img), type=2, norm='ortho')
    dct_img = np.log(np.abs(dct_img) + eps)
    return dct_img


def compute_fingerprint(genuine_paths, synthetic_paths):
    genuine_spectra = [compute_peak_dct(p) for p in tqdm(genuine_paths, desc="Genuine DCTs")]
    synthetic_spectra = [compute_peak_dct(p) for p in tqdm(synthetic_paths, desc="Synthetic DCTs")]

    mean_genuine = np.mean(genuine_spectra, axis=0)
    mean_synthetic = np.mean(synthetic_spectra, axis=0)

    return np.expm1(mean_synthetic - mean_genuine)


def inverse_dct(dct_coeffs):
    return idctn(dct_coeffs, type=2, norm='ortho')

def scale_fp_to_01(fp):
    min_val = np.min(fp)
    max_val = np.max(fp)
    return (fp - min_val) / (max_val - min_val + 1e-8)

def apply_peak_spectrum_attack(image_path, fingerprint, strength=10.0):
    original_dct = compute_dct(image_path)
    fingerprint = scale_fp_to_01(fingerprint)
    attacked_dct = original_dct * (1 - strength * fingerprint)
    attacked_img = inverse_dct(attacked_dct)
    attacked_img = np.clip(attacked_img, 0, 255).astype(np.uint8)
    return attacked_img

def process_and_save_attacks(input_dir, output_dir, fingerprint, strength=1.0):
    image_paths = glob.glob(os.path.join(input_dir, '**/*.png'), recursive=True)

    for path in tqdm(image_paths, desc=f"Attacking {os.path.basename(input_dir)}"):
        rel_path = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            attacked_img = apply_peak_spectrum_attack(path, fingerprint, strength)
            cv2.imwrite(out_path, attacked_img)
        except Exception as e:
            print(f"Error processing {path}: {e}")

# === Main Script ===

#genuine_paths = glob.glob(r'images/genuine/idiap/**/*.png', recursive=True)
genuine_paths = glob.glob(r'images/genuine/idiap/**/*.png', recursive=True)
synthetic_root = r'images/synthetic/idiap'
output_root = r'peak_output_10/synthetic_attacked/idiap'
synthetic_methods = ['cycleGAN', 'distanceGAN', 'dritGAN','starGAN']  # Add others here

for method in synthetic_methods:
    print(f"\n=== Processing method: {method} ===")

    synthetic_dir = os.path.join(synthetic_root, method)
    output_dir = os.path.join(output_root, method)
    synthetic_paths = glob.glob(os.path.join(synthetic_dir, '**/*.png'), recursive=True)

    if not synthetic_paths:
        print(f"No images found for method: {method}")
        continue

    fingerprint = compute_fingerprint(genuine_paths, synthetic_paths)
    process_and_save_attacks(synthetic_dir, output_dir, fingerprint)
