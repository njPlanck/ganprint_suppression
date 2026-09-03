import sys
import types
import collections.abc
import torch
import torch.nn as nn

# --- 1. THE MONKEY PATCH (Must be before 'from tinynet import ...') ---
if not hasattr(torch, "_six"):
    six_module = types.ModuleType("_six")
    six_module.container_abcs = collections.abc
    sys.modules["torch._six"] = six_module

# --- 2. REGULAR IMPORTS ---
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

# --- 3. THE MODEL IMPORT ---
from python_projects.ganprint_removal.feat_extraction.tinynet import tinynet

def load_trained_model(model_path, model_type='tinynet_e', device='cpu'):
    configs = {
        'tinynet_a': {'r': 0.86, 'w': 1.0, 'd': 1.2},
        'tinynet_e': {'r': 0.475, 'w': 0.51, 'd': 0.60},
    }
    cfg = configs[model_type]
    
    # 1. Initialize original skeleton (3-channel, 1000 classes)
    model = tinynet(r=cfg['r'], w=cfg['w'], d=cfg['d'], num_classes=1000, in_chans=3)
    
    # 2. Modify Stem for 1-channel Grayscale (Matches training script)
    old_conv = model.conv_stem
    model.conv_stem = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                                stride=old_conv.stride, padding=old_conv.padding, bias=False)
    
    # 3. Modify Classifier for 3 classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)
    
    # 4. Load the weights
    print(f">>> Loading PyTorch weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 5. Isolate Feature Extractor
    model.classifier = nn.Identity()
    
    return model, cfg['r']

def extract_features_from_dir(directory, model, transform_fn, device):
    feats = []
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return feats
        
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                continue
            
            path = os.path.join(root, file)
            img = Image.open(path).convert('L')
            img_t = transform_fn(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat_vec = model(img_t).cpu().numpy().flatten()
                feats.append(feat_vec)
                
    return feats

def extract_features(model_path, genuine_dir, spoofed_dir, synthetic_dir, output_csv, model_type='tinynet_e'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, res_scale = load_trained_model(model_path, model_type, device)
    
    # Preprocessing
    input_res = int(224 * res_scale)
    transform_fn = transforms.Compose([
        transforms.Resize((input_res, input_res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), 
    ])

    # 1. Process Features exactly for the provided paths
    print(f">>> Scanning Genuine directory: {genuine_dir}")
    genuine_feats = extract_features_from_dir(genuine_dir, model, transform_fn, device)
    
    print(f">>> Scanning Spoofed directory: {spoofed_dir}")
    spoofed_feats = extract_features_from_dir(spoofed_dir, model, transform_fn, device)

    print(f">>> Scanning Synthetic directory: {synthetic_dir}")
    synthetic_feats = extract_features_from_dir(synthetic_dir, model, transform_fn, device)

    # Convert to arrays
    gen_arr = np.array(genuine_feats)
    spf_arr = np.array(spoofed_feats)
    syn_arr = np.array(synthetic_feats)

    if len(gen_arr) == 0 or len(spf_arr) == 0 or len(syn_arr) == 0:
        print("Warning: One of your directories is empty. Check your data paths.")
        return

    # Build Table: 0=Genuine, 1=Spoofed, 2=Synthetic (GAN)
    rows_0 = np.column_stack((np.zeros(len(gen_arr)), gen_arr))
    rows_1 = np.column_stack((np.ones(len(spf_arr)), spf_arr))
    rows_2 = np.column_stack((np.full(len(syn_arr), 2), syn_arr))
    
    final_table = np.vstack((rows_0, rows_1, rows_2))
    df = pd.DataFrame(final_table)
    
    # Name columns: label, bin_0, bin_1, ...
    df.columns = ['label'] + [f'bin_{i}' for i in range(final_table.shape[1]-1)]
    
    # Save directly to the single CSV path specified
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Generated: {output_csv} ({len(final_table)} samples)")

if __name__ == "__main__":
    extract_features(
        model_path='./model/tinynet_vein_scut.pth', 
        genuine_dir='../../auto_encoder/outputs_impainting/images/scut_fft/starGAN/genuine',
        spoofed_dir='../../auto_encoder/outputs_impainting/images/scut_fft/starGAN/spoofed',
        # NOTE: Added "/cycleGAN" to this path so it only targets that specific folder
        synthetic_dir='../../pad_dataset/scut/synthetic/starGAN', 
        # NOTE: Changed from a directory parameter to a direct CSV file path
        output_csv='../../auto_encoder/outputs_impainting/csvs/scut_fft/starGAN/tinynet_starGAN.csv',
        model_type='tinynet_e'
    )