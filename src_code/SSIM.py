#!/usr/bin/env python
# coding: utf-8


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ----------- Helper Functions -----------

def get_image_files(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg','.bmp'))])

def compute_ssim_for_folders(reference_folder, compare_folder):
    ref_files = get_image_files(reference_folder)
    comp_files = get_image_files(compare_folder)

    common_files = sorted(set(ref_files) & set(comp_files))
    if not common_files:
        print(f"No common images to compare between:\n{reference_folder}\nand\n{compare_folder}\n")
        return [], [], 0.0

    ssim_scores = []

    for file in common_files:
        img1_path = os.path.join(reference_folder, file)
        img2_path = os.path.join(compare_folder, file)

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Could not read image {file}, skipping.")
            continue

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        score, _ = ssim(img1, img2, full=True)
        ssim_scores.append(score)

    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0
    return common_files, ssim_scores, avg_ssim

# ----------- Folder Setup -----------

def smooth_curve(y, window_size=1):
    return np.convolve(y, np.ones(window_size)/window_size, mode='same')

reference_folder = r"/home/chinasa/python_projects/denoising/images/idiap/synthetic/cycleGAN"


attack_folders = {
    "IDIAP cycleGAN": r"/home/chinasa/python_projects/auto_encoder/outputs/images/idiap2/cycleGAN"
}



# ----------- Compute SSIM and Collect Data -----------

all_ssim_data = {}

for attack_name, attack_folder in attack_folders.items():
    print(f"Processing: {attack_name}")
    files, scores, avg = compute_ssim_for_folders(reference_folder, attack_folder)
    all_ssim_data[attack_name] = (files, scores, avg)

# ----------- Plotting -----------
'''
plt.figure(figsize=(12, 6))

for attack_name, (files, scores, avg) in all_ssim_data.items():
    plt.plot(range(len(scores)), scores, marker='o', label=f"{attack_name} (avg: {avg:.4f})")

plt.title('SSIM Scores: Attacked vs. Reference distancegan Images')
plt.xlabel('Image Index')
plt.ylabel('SSIM Score')
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''

plt.style.use("seaborn-v0_8-darkgrid")
plt.figure(figsize=(12, 6))

for attack_name, (files, scores, avg) in all_ssim_data.items():
    smoothed = smooth_curve(scores, window_size=5)
    plt.plot(smoothed, linewidth=2, marker='', label=f"{attack_name} (avg: {avg:.4f})")

plt.title('SSIM Scores for Denoised on cycleGAN Images', fontsize=16, fontweight='bold')
plt.xlabel('Image Index', fontsize=12)
plt.ylabel('SSIM Score', fontsize=12)
plt.ylim(0, 1.05)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("idiapcycle.png")
