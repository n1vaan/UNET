import torch
from torch import nn

class Block(nn.Module):

    ''' Encoder and Decoder Blocks: 
        Encoders are double Conv2d() and ReLU layers. 
        Decoders are double Conv2d() and ReLU layers followed by a ConvTranspose2d() layer for upsampling. '''
    
    def __init__(self, type, in_channels:int, out_channels:int):
        super().__init__()
        if type == "enc": 
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3)),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3)),
                nn.ReLU()
            )
        else: 
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3)),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3)),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(2,2),stride=2),
            )
    def forward(self, X):
        return self.block(X)
    