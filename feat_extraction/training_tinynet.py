
# TOP OF training_tinynet.py
import sys
import collections.abc
import types
import torch

# Create the fake module before any other imports
if not hasattr(torch, "_six"):
    six_module = types.ModuleType("_six")
    six_module.container_abcs = collections.abc
    sys.modules["torch._six"] = six_module

# Now continue with your other imports...
from python_projects.ganprint_removal.feat_extraction.tinynet import tinynet

# NOW you can do your regular imports
import torch
import torch.nn as nn
# ... the rest of your imports
from python_projects.ganprint_removal.feat_extraction.tinynet import tinynet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import argparse
import os

from python_projects.ganprint_removal.feat_extraction.tinynet import tinynet 

# --- Helper to handle independent transforms for splits ---
class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

class VeinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        class_map = {'genuine': 0, 'spoofed': 1, 'synthetic': 2}
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

        for category, label_idx in class_map.items():
            cat_path = os.path.join(root_dir, category)
            if not os.path.exists(cat_path):
                continue
                
            for root, _, files in os.walk(cat_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        self.images.append(os.path.join(root, file))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Open as grayscale for your vein requirements
        image = Image.open(img_path).convert('L') 
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main(args):
    configs = {
        'tinynet_a': {'r': 0.86, 'w': 1.0, 'd': 1.2},
        'tinynet_e': {'r': 0.475, 'w': 0.51, 'd': 0.60},
    }

    # Force architecture to match weight file if 'tinynet_e' is in the path
    if 'tinynet_e' in args.pretrained_path.lower():
        args.model_name = 'tinynet_e'
        print(">>> Automatic Detection: Using TinyNet-E configuration.")

    cfg = configs.get(args.model_name, configs['tinynet_e'])
    print(f">>> Model: {args.model_name} (Resolution: {cfg['r']}, Width: {cfg['w']})")

    # 1. Initialize Model
    # 1. Initialize Model with standard 3-channel input for weight loading
    model = tinynet(r=cfg['r'], w=cfg['w'], d=cfg['d'], num_classes=1000, in_chans=3)

    # 2. Load Pre-trained Weights (BEFORE modifying layers)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f">>> Loading weights from {args.pretrained_path}...")
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        
        # We load what we can. The classifier won't match if it's already 3, 
        # but here we initialized it as 1000 to ensure a clean load.
        model.load_state_dict(state_dict, strict=True) 

    # 3. NOW Modify Stem for 1-channel Grayscale
    old_conv = model.conv_stem
    model.conv_stem = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                                stride=old_conv.stride, padding=old_conv.padding, bias=False)
    
    # Optional: Initialize the new 1-channel conv with the average of the 3 RGB channels
    with torch.no_grad():
        model.conv_stem.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)

    # 4. NOW Modify Classifier for 3 classes (Vein categories)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4. Transforms
    input_res = int(224 * cfg['r'])
    transform_train = transforms.Compose([
        transforms.Resize((input_res, input_res)),
        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), 
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((input_res, input_res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 5. Data Splitting
    full_dataset = VeinDataset(args.data, transform=None) # No transform at base level
    if len(full_dataset) == 0:
        print("Error: No images found.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # Wrap subsets with independent transforms
    train_loader = DataLoader(ApplyTransform(train_subset, transform_train), 
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ApplyTransform(val_subset, transform_val), 
                            batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f">>> Split Complete: {train_size} Training | {val_size} Validation")

    # 6. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 7. Loop
    best_acc = 0.0
    os.makedirs('./model', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = train_loss / train_size

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / val_size
        val_acc = 100. * correct / val_size

        print(f"Epoch [{epoch+1:03d}/{args.epochs}] "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./model/tinynet_vein_idiap.pth")
            print(f"--- Best Model Saved ({best_acc:.2f}%) ---")
        
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='tinynet_e')
    parser.add_argument('--pretrained_path', type=str, default='./model/tinynet_e.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    main(args)