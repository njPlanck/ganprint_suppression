import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from auto_encoder import FingerprintDenoisingAE


model_path = "/home/chinasa/python_projects/auto_encoder/outputs/scut_fingerprint_denoiser.pth"
input_root = "/home/chinasa/python_projects/pad_dataset/scut/synthetic/starGAN"
output_root = "/home/chinasa/python_projects/auto_encoder/outputs/images/scut_fft/starGAN"

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # Normalize [-1, 1]
])

def denormalize(tensor):
    """Reverse normalization from [-1, 1] -> [0, 1] for saving"""
    return tensor * 0.5 + 0.5

#load model
print("Loading model...")

model = FingerprintDenoisingAE().to(device)

# Load weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded successfully.")

valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Ensure output root exists
os.makedirs(output_root, exist_ok=True)

with torch.no_grad():
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            if not filename.lower().endswith(valid_exts):
                continue
            
            # Construct input path
            input_path = os.path.join(root, filename)
            
            # Construct output path 
            relative_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)

            # Load and Preprocess
            try:
                # Open as RGB, Transform will convert to Grayscale(1)
                img = Image.open(input_path).convert("RGB") 
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Denoise
                output_tensor = model(img_tensor)

                # Save
                output_img = denormalize(output_tensor.squeeze(0))
                save_image(output_img, output_path)

                print(f"Denoised: {relative_path}/{filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print(f"\n Processing complete! Results saved in: {output_root}")