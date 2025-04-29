import torch
import torch.nn as nn
import torch.nn.functional as F
from models.msfgm import MSFGM

class ResNetBlock(nn.Module):
    """
    Basic ResNet block for 1D signals
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_msfgm=False):
        """
        Initialize ResNet block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            downsample: Optional downsampling module
            use_msfgm: Whether to use MSFGM
        """
        super(ResNetBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution or MSFGM
        if use_msfgm:
            self.conv2 = MSFGM(out_channels)
        else:
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.use_msfgm = use_msfgm

    def forward(self, x):
        """
        Forward pass through ResNet block
        
        Args:
            x: Input tensor [batch_size, in_channels, length]
            
        Returns:
            Output tensor [batch_size, out_channels, length]
        """
        identity = x
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution or MSFGM
        if self.use_msfgm:
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
        
        # Apply downsampling if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    """
    ResNet backbone for 1D signals
    """
    def __init__(self, input_channels, output_size, hidden_channels, num_blocks=20, use_msfgm=True):
        """
        Initialize ResNet
        
        Args:
            input_channels: Number of input channels
            output_size: Dimension of output
            hidden_channels: Number of hidden channels
            num_blocks: Number of ResNet blocks
            use_msfgm: Whether to use MSFGM in ResNet blocks
        """
        super(ResNet, self).__init__()
        self.in_channels = hidden_channels
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet blocks
        self.layer1 = self._make_layer(hidden_channels, hidden_channels, num_blocks, use_msfgm)
        
        # Output layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_channels, output_size)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, use_msfgm):
        """
        Create a layer of ResNet blocks
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of blocks in the layer
            use_msfgm: Whether to use MSFGM
        
        Returns:
            Sequential layer of ResNet blocks
        """
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNetBlock(in_channels, out_channels, use_msfgm=use_msfgm))
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through ResNet
        
        Args:
            x: Input tensor [batch_size, input_channels, length]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Output layer
        x = self.fc_out(x)
        
        return x