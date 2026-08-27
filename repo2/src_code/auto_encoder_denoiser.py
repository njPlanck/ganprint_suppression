import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class FingerprintDenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2), # Downsample

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4 (Bottleneck Entrance)
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Bottleneck(Spatial)
        self.bottleneck = nn.Sequential(
            # FIX: padding=0 for kernel_size=1
            nn.Conv2d(256, 128, 1, padding=0, bias=False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 64, 1, padding=0, bias=False), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Keep padding=1 here because kernel_size=3
            nn.Conv2d(64, 64, 3, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 1, padding=0, bias=False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 1, padding=0, bias=False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Up 1
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Up 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Up 3
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Up 4
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final reconstruction
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        target_h, target_w = x.size(2), x.size(3)
        
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        
        # Handle strict dimension matching for odd input sizes (e.g., 665x220)
        if x.size(2) != target_h or x.size(3) != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
        return x