import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, leaky_relu_slope=0.01):
        super().__init__()
        self.is_downsample = (input_channel != output_channel)

        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=input_channel),
            nn.LeakyReLU(leaky_relu_slope,inplace=True),
            nn.MaxPool2d(kernel_size=(1,4))
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=output_channel,
                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(output_channel,output_channel,3,padding=1,bias=False)
        )

        self.conv_1by1 = None
        if self.is_downsample:
            self.conv_1by1 = nn.Conv2d(input_channel,output_channel,1,bias=False)
    
    def forward(self,x):
        x = self.pre_conv(x)
        if self.is_downsample:
            x = self.conv(x) + self.conv_1by1(x)
        else:
            x = self.conv(x) + x
        return x