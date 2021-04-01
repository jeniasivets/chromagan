import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, channels_from, channels_to, kernel_size, stride, padding, bn=True, slope=0):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels_from, channels_to, kernel_size, stride, padding))
        if bn:
            self.layers.add_module('bn', nn.BatchNorm2d(channels_to))
        self.layers.add_module('act', nn.LeakyReLU(slope))
        
        
    def forward(self, x):
        return self.layers(x)
     

class Generator_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Sequential(
            Block(512, 512, kernel_size=3, stride=1, padding=1),
            Block(512, 256, kernel_size=3, stride=1, padding=1)
        )
    
        self.layers_2 = nn.Sequential(
            Block(512, 256, kernel_size=3, stride=1, padding=1, bn=False),
            Block(256, 128, kernel_size=3, stride=1, padding=1, bn=False),
            nn.Upsample(scale_factor=2),
            Block(128, 64, kernel_size=3, stride=1, padding=1, bn=False),
            Block(64, 64, kernel_size=3, stride=1, padding=1, bn=False),
            nn.Upsample(scale_factor=2),
            Block(64, 32, kernel_size=3, stride=1, padding=1, bn=False),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2)
        )
    
    def forward(self, x, gen2_out):
        x = self.layers_1(x)
        new_channels = gen2_out[:, :, None, None].repeat(1, 1, 28, 28)
        x = torch.cat([x, new_channels], dim=1)
        return self.layers_2(x)
    
    
class Generator_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Block(512, 512, kernel_size=3, stride=2, padding=1),
            Block(512, 512, kernel_size=3, stride=1, padding=1),
            Block(512, 512, kernel_size=3, stride=2, padding=1),
            Block(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        self.output_1 = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
        )

        self.output_2 = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 200),
        )
    def forward(self, x):
        x = self.layers(x)
        return self.output_1(x), self.output_2(x)
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Block(3, 64, kernel_size=4, stride=2, padding=1, bn=False, slope=0.2),
            Block(64, 128, kernel_size=4, stride=2, padding=1, slope=0.2),
            Block(128, 256, kernel_size=4, stride=2, padding=1, slope=0.2),
            nn.ZeroPad2d((1,0,1,0)),
            Block(256, 512, kernel_size=4, stride=1, padding=1, slope=0.2),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)
    
    
class Generator(nn.Module):
    def __init__(self, vgg16):
        super().__init__()
        self.vgg16 = vgg16
        self.gen_1 = Generator_1()
        self.gen_2 = Generator_2()
        
    def forward(self, x):
        vgg_preds = self.vgg16(x)
        out_1, out_2 = self.gen_2(vgg_preds)
        out = self.gen_1(vgg_preds, out_1)
        return out, out_2

