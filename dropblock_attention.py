import torch
import torch.nn.functional as F
from torch import nn

class DropBlock2D(nn.Module):
    def __init__(self, keep_prob=0.9, block_size=7, beta=0.9):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.beta = beta
        
    def normalize(self, input):
        min_c, max_c = input.min(1, keepdim=True)[0], input.max(1, keepdim=True)[0]
        input_norm = (input - min_c) / (max_c - min_c + 1e-8)
        return input_norm
        
    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
            
        # Get spatial dimensions
        _, _, height, width = input.shape
        
        # Adjust block_size if needed
        block_size = min(self.block_size, min(height, width))
        
        # Calculate gamma with safeguards against division by zero
        gamma = (1. - self.keep_prob) / (block_size ** 2)
        for sh in [height, width]:
            denom = max(1, sh - block_size + 1)  # Ensure denominator is never zero
            gamma *= sh / denom
            
        # Create bernoulli mask
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        
        # Create convolution kernel for block masking
        kernel = torch.ones((input.shape[1], 1, block_size, block_size)).to(
            device=input.device, dtype=input.dtype
        )
        
        # Apply convolution to create block mask
        Msum = F.conv2d(M, kernel, padding=block_size // 2, groups=input.shape[1])
        Msum = (Msum < 1).to(device=input.device, dtype=input.dtype)
        
        # Apply mask and normalization
        input2 = input * Msum
        x_norm = self.normalize(input2)
        mask = (x_norm > self.beta).float()
        block_mask = 1 - (mask * x_norm)
        
        return input * block_mask
