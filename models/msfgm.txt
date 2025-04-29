import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import HybridAttentionModule

class MSFGM(nn.Module):
    """
    Multi-Scale Fusion Gated Module
    """
    def __init__(self, channels, s=4, use_attention=True):
        """
        Initialize MSFGM
        
        Args:
            channels: Number of input/output channels
            s: Number of feature subsets
            use_attention: Whether to use hybrid attention
        """
        super(MSFGM, self).__init__()
        self.channels = channels
        self.s = s
        self.use_attention = use_attention
        
        # Ensure channels is divisible by s
        assert channels % s == 0, f"Channels ({channels}) must be divisible by s ({s})"
        
        # Convolution blocks for each subset (except the first one)
        self.convs = nn.ModuleList()
        for i in range(1, s):
            self.convs.append(nn.Conv1d(
                channels // s, channels // s, 
                kernel_size=3, padding=1
            ))
        
        # Gating modules
        self.gates = nn.ModuleList()
        for i in range(1, s):
            self.gates.append(nn.Sequential(
                nn.Conv1d(channels, 1, kernel_size=1),
                nn.Sigmoid()
            ))
        
        # 1x1 conv for fusion
        self.fusion = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Hybrid attention module
        if use_attention:
            self.attention = HybridAttentionModule(channels)
        
    def forward(self, x):
        """
        Forward pass through MSFGM
        
        Args:
            x: Input tensor [batch_size, channels, length]
            
        Returns:
            Enhanced multi-scale features
        """
        # Split input into s subsets along channel dimension
        x_splits = torch.chunk(x, self.s, dim=1)
        
        # Process first subset directly (no convolution)
        outputs = [x_splits[0]]
        
        # Process remaining subsets with gating
        for i in range(1, self.s):
            if i == 1:
                # For second subset, apply convolution directly
                y_i = self.convs[i-1](x_splits[i])
            else:
                # For subsequent subsets, add gated previous output
                g_i = self.gates[i-1](x)
                y_i = self.convs[i-1](x_splits[i] + g_i * outputs[-1])
            
            outputs.append(y_i)
        
        # Concatenate all outputs
        y_concat = torch.cat(outputs, dim=1)
        
        # Apply 1x1 conv and residual connection
        y_fused = self.fusion(y_concat)
        y_res = y_fused + x
        
        # Apply hybrid attention if enabled
        if self.use_attention:
            y_final = self.attention(y_res)
        else:
            y_final = y_res
        
        return y_final