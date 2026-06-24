import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        
        
        #Building the residual sequence
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                            3, stride, padding=1, bias=False),
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                            3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
        
    def forward(self, x):
        
        return x