import torch
from torch import nn
import torch.nn.functional as F
from Block import Block

class UNET(nn.Module):

    '''UNET based on design from original paper: https://arxiv.org/abs/1505.04597'''

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        # top down 
        self.enc1 = Block(type="enc", in_channels=in_channels, out_channels=64)
        self.enc2 = Block(type="enc", in_channels=64, out_channels=128)
        self.enc3 = Block(type="enc", in_channels=128, out_channels=256)
        self.enc4 = Block(type="enc", in_channels=256, out_channels=512)

        # bottom up
        self.dec1 = Block(type="dec", in_channels=512, out_channels=1024)
        self.dec2 = Block(type="dec", in_channels=1536, out_channels=512)
        self.dec3 = Block(type="dec", in_channels=768, out_channels=256)
        self.dec4 = Block(type="dec", in_channels=384, out_channels=128)

        self.exit = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(1,1))
        )
    
    def concatenate(self, enc, dec):

        enc_shape = enc.shape[1:]
        dec_shape = dec.shape[1:]

        crop_h = (enc_shape[0] - dec_shape[0]) // 2
        crop_w = (enc_shape[1] - dec_shape[1]) // 2
        
        enc_cropped = enc[:, crop_h:crop_h + dec_shape[0], crop_w:crop_w + dec_shape[1]]
        
        return torch.cat([enc_cropped, dec], dim=0)
    

    def forward(self, X):

        enc1 = self.enc1(X)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        res1 = self.dec1(F.max_pool2d(enc4, 2))
        res1 = self.concatenate(enc4, res1)

        res2 = self.dec2(res1)
        res2 = self.concatenate(enc3, res2)

        res3 = self.dec3(res2)
        res3 = self.concatenate(enc2, res3)

        res4 = self.dec4(res3)
        res4 = self.concatenate(enc1, res4)

        res5 = self.exit(res4)

        return res5
    
