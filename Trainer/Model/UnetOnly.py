import torch.nn as nn
import torch
from .Unet import Unet
class UnetOnly(nn.Module):
    def __init__(self,device):
        super(UnetOnly,self).__init__()
        self.device = device
        self.unet = Unet(input_channel=2)
        self.conv_block = nn.Sequential(
                                        nn.Conv2d(in_channels=1,out_channels=2,kernel_size=1,bias=False),
                                        nn.BatchNorm2d(num_features=2),
                                        nn.LeakyReLU(0.01, inplace=True)
        )
        self.final_activaion = nn.Sigmoid()
    
    def forward(self,x):
        padding = torch.zeros(x.shape[0],x.shape[1],1,x.shape[3]).to(self.device)
        x= torch.cat([x,padding],dim=2)
        x=self.conv_block(x)
        mask_vocal, mask_accom = self.unet(x)
        mask_vocal = self.final_activaion(mask_vocal)
        mask_accom = self.final_activaion(mask_accom)
        return mask_vocal[:,:,:31,:] , mask_accom[:,:,:31,:]