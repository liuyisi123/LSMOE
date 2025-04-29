import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttentionModule(nn.Module):
    """
    Hybrid Attention Module combining channel and spatial attention
    """
    def __init__(self, channels, num_groups=32):
        """
        Initialize Hybrid Attention Module
        
        Args:
            channels: Number of input channels
            num_groups: Number of groups for feature division
        """
        super(HybridAttentionModule, self).__init__()
        self.channels = channels
        self.num_groups = num_groups
        
        # Group normalization for spatial attention
        self.group_norm = nn.GroupNorm(min(num_groups, channels), channels)
        
        # Channel attention parameters
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            ) for _ in range(num_groups)
        ])
        
        # Spatial attention parameters
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            ) for _ in range(num_groups)
        ])
        
    def forward(self, x):
        """
        Forward pass through the hybrid attention module
        
        Args:
            x: Input tensor of shape [batch_size, channels, length]
            
        Returns:
            Attention-enhanced features of same shape as input
        """
        # Get dimensions
        batch_size, C, L = x.shape
        
        # Split features into groups
        x_groups = x.view(batch_size, self.num_groups, C // self.num_groups, L)
        
        # Process each group
        output_groups = []
        for g in range(self.num_groups):
            # Get current group
            x_g = x_groups[:, g, :, :]  # [batch_size, C//G, L]
            
            # Split into two branches for channel and spatial attention
            c_per_group = x_g.size(1)
            x_g1, x_g2 = torch.split(x_g, c_per_group // 2, dim=1)
            
            # Channel attention: global average pooling across spatial dimension
            channel_avg = x_g1.mean(dim=2, keepdim=True)  # [batch_size, C//2G, 1]
            channel_weights = self.channel_attention[g](channel_avg)
            x_g1_enhanced = x_g1 * channel_weights
            
            # Spatial attention: normalize and compute attention weights
            x_g2_norm = self.group_norm(x_g2)
            spatial_weights = self.spatial_attention[g](x_g2_norm.mean(dim=1, keepdim=True).transpose(1, 2))
            x_g2_enhanced = x_g2 * spatial_weights.transpose(1, 2)
            
            # Concatenate enhanced features
            x_g_enhanced = torch.cat([x_g1_enhanced, x_g2_enhanced], dim=1)
            output_groups.append(x_g_enhanced)
        
        # Concatenate all groups
        output = torch.cat(output_groups, dim=1).view(batch_size, C, L)
        
        # Channel shuffle operation for information exchange
        output = output.view(batch_size, self.num_groups, C // self.num_groups, L)
        output = output.transpose(1, 2).contiguous().view(batch_size, C, L)
        
        return output