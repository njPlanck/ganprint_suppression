import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pytorch_msssim import MS_SSIM
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from auto_encoder import FingerprintDenoisingAE

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")

#LOCAL paths
data_dir = "/home/chinasa/python_projects/denoising/images/scut/spoofed" 
output_root = "/home/chinasa/python_projects/auto_encoder/outputs"
os.makedirs(output_root, exist_ok=True)
save_path = os.path.join(output_root, "scut_fingerprint_denoiser2.pth")
sample_dir = os.path.join(output_root, 'samples')
os.makedirs(sample_dir, exist_ok=True)

# Hyperparameters
batch_size = 4  
epochs = 150
lr = 1e-4 
weight_decay = 1e-5

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

try:
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
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

# --- 5. Setup Training ---
model = FingerprintDenoisingAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#scaler = GradScaler() # Initialize Mixed Precision Scaler

# Loss: L1 for pixel accuracy, SSIM for structural integrity (ridges)
l1_criterion = nn.L1Loss()
ssim_criterion = MS_SSIM(data_range=2.0, size_average=True, channel=1)

lambda_l1 = 0.8
lambda_ssim = 0.2

# training Loop with Mixed Precision ---
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
        loss = (lambda_l1 * l1) + (lambda_ssim * ssim)

        loss.backward() 
        optimizer.step() 
        
        train_loss += loss.item() * imgs.size(0)
        
    train_loss /= len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            
            l1 = l1_criterion(outputs, imgs)
            ssim = 1.0 - ssim_criterion(outputs, imgs)
            loss = (lambda_l1 * l1) + (lambda_ssim * ssim)
            
            val_loss += loss.item() * imgs.size(0)
            
    val_loss /= len(val_loader.dataset)

    # --- Visualization ---
    if (epoch + 1) % 5 == 0:
        # take the first batch from val_loader for visualization
        fixed_imgs, _ = next(iter(val_loader))
        fixed_imgs = fixed_imgs.to(device)
        with torch.no_grad():
            recon = model(fixed_imgs)
            # Only save the first 4 images to save space
            comparison = torch.cat((fixed_imgs[:4], recon[:4]), dim=0)
            save_image(comparison * 0.5 + 0.5, f"{sample_dir}/epoch_{epoch+1}.png", nrow=4)

    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}: New best model saved (Val Loss: {val_loss:.5f})")
    else:
      print(f"Epoch {epoch+1}: Train: {train_loss:.5f} | Val: {val_loss:.5f}")