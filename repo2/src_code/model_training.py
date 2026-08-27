import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
from pytorch_msssim import MS_SSIM
from auto_encoder import FingerprintDenoisingAE

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")

class SpectralLoss(nn.Module):
    def __init__(self, use_log=True, eps=1e-8):
        super(SpectralLoss, self).__init__()
        self.use_log = use_log
        self.eps = eps

    def forward(self, pred, target):
        # Compute 2D Real FFT across spatial dimensions (H, W)
        fft_pred = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        fft_target = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        
        # Extract magnitudes
        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)
        
        # Log scaling prevents center DC component from dominating
        if self.use_log:
            mag_pred = torch.log(mag_pred + self.eps)
            mag_target = torch.log(mag_target + self.eps)
            
        # L1 loss on the frequency magnitude spectra
        return F.l1_loss(mag_pred, mag_target)


# LOCAL paths
data_dir = "/home/chinasa/python_projects/pad_dataset/scut/spoofed" 
output_root = "/home/chinasa/python_projects/auto_encoder/outputs"
os.makedirs(output_root, exist_ok=True)
save_path = os.path.join(output_root, "scut_fingerprint_denoiser.pth")
sample_dir = os.path.join(output_root, 'samples_denoised')
os.makedirs(sample_dir, exist_ok=True)

# Hyperparameters
batch_size = 4  
epochs = 100
lr = 1e-4 
weight_decay = 1e-5

class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Grabs all png and jpg files directly in the cycleGAN folder
        self.image_paths = glob.glob(os.path.join(root_dir, '**', '*.png')) + \
                           glob.glob(os.path.join(root_dir, '**', '*.jpg')) + \
                           glob.glob(os.path.join(root_dir, '**', '*.bmp'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L") # Ensure grayscale
        if self.transform:
            image = self.transform(image)
        return image, 0 # Return 0 as a dummy label since we don't need classes


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

try:
    dataset = FlatImageDataset(root_dir=data_dir, transform=transform)
    print(f"Dataset found: {len(dataset)} images")
except FileNotFoundError:
    print(f"Error: Create folder structure: {data_dir}/class_name/images.png")
    exit()

# Split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

# Setup Training
model = FingerprintDenoisingAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# INITIALIZE LOSS FUNCTIONS & WEIGHTS

l1_criterion = nn.L1Loss()
ssim_criterion = MS_SSIM(data_range=2.0, size_average=True, channel=1)
spectral_criterion = SpectralLoss(use_log=True)

lambda_l1 = 0.75
lambda_ssim = 0.15
lambda_freq = 0.10  # Weight for the new spectral loss

# Training Loop
best_val_loss = float('inf')

print("Starting local training...")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(imgs)
            
        l1 = l1_criterion(outputs, imgs)
        ssim = 1.0 - ssim_criterion(outputs, imgs)
        freq = spectral_criterion(outputs, imgs)  # Compute frequency loss
        
        # Combined Loss
        loss = (lambda_l1 * l1) + (lambda_ssim * ssim) + (lambda_freq * freq)

        loss.backward() 
        optimizer.step() 
        
        train_loss += loss.item() * imgs.size(0)
        
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            
            l1 = l1_criterion(outputs, imgs)
            ssim = 1.0 - ssim_criterion(outputs, imgs)
            freq = spectral_criterion(outputs, imgs)  # Compute validation frequency loss
            
            loss = (lambda_l1 * l1) + (lambda_ssim * ssim) + (lambda_freq * freq)
            
            val_loss += loss.item() * imgs.size(0)
            
    val_loss /= len(val_loader.dataset)

    # Visualization
    if (epoch + 1) % 5 == 0:
        fixed_imgs, _ = next(iter(val_loader))
        fixed_imgs = fixed_imgs.to(device)
        with torch.no_grad():
            recon = model(fixed_imgs)
            comparison = torch.cat((fixed_imgs[:4], recon[:4]), dim=0)
            save_image(comparison * 0.5 + 0.5, f"{sample_dir}/epoch_{epoch+1}.png", nrow=4)

    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}: New best model saved (Val Loss: {val_loss:.5f})")
    else:
        print(f"Epoch {epoch+1}: Train: {train_loss:.5f} | Val: {val_loss:.5f}")