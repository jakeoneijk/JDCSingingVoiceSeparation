import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    
    def __init__(self,input_channel):
        super(Unet, self).__init__()
        self.input_channel = input_channel
        self.next_channel = self.input_channel * 2

        self.encoding_list = nn.ModuleList()
        for i in range(8):
            if i < 4:
                self.encoding_list.append(self.encoding_conv_freq(self.input_channel , self.next_channel))
            else:
                self.encoding_list.append(self.encoding_conv(self.input_channel , self.next_channel))
            self.input_channel_size_update_encoding()

        self.next_channel = int(self.next_channel / 4)

        self.decoding_vocal_list = nn.ModuleList()
        self.decoding_accom_list = nn.ModuleList()
        for i in range(7):
            if i < 3:
                self.decoding_vocal_list.append(self.decoding_conv(self.input_channel,self.next_channel,True))
                self.decoding_accom_list.append(self.decoding_conv(self.input_channel,self.next_channel,True))
            elif i == 3:
                self.decoding_vocal_list.append(self.decoding_conv(self.input_channel,self.next_channel,False))
                self.decoding_accom_list.append(self.decoding_conv(self.input_channel,self.next_channel,False))
            else:
                self.decoding_vocal_list.append(self.decoding_conv_freq(self.input_channel,self.next_channel,False))
                self.decoding_accom_list.append(self.decoding_conv_freq(self.input_channel,self.next_channel,False))
            self.input_channel_size_update_decoding()

        self.next_channel = int(self.next_channel / 2)

        self.decoding_vocal_list.append(self.decoding_conv_freq(self.input_channel,self.next_channel,False))
        self.decoding_accom_list.append(self.decoding_conv_freq(self.input_channel,self.next_channel,False))
    
    def forward(self , x):
        encoding_feature_list = []
        for i in range(8):
            x = self.encoding_list[i](x)
            encoding_feature_list.append(x)

        decoding_vocal_feature = self.decoding_vocal_list[0](encoding_feature_list[7])
        decoding_accom_feature = self.decoding_accom_list[0](encoding_feature_list[7])
        for i in range(1,8):
            decoding_vocal_feature = self.decoding_vocal_list[i](torch.cat([decoding_vocal_feature,encoding_feature_list[7-i]],dim=1))
            decoding_accom_feature = self.decoding_accom_list[i](torch.cat([decoding_accom_feature,encoding_feature_list[7-i]],dim=1))

        return decoding_vocal_feature , decoding_accom_feature
    

    def encoding_conv(self,input_channel , output_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=8,stride=2,padding=3),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2)
        )

    def encoding_conv_freq(self,input_channel , output_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=(1,8),stride=(1,2),padding=(0,3)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2)
        )
    
    def decoding_conv(self,input_channel , output_channel , use_dropout):
        if use_dropout:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channel , output_channel , kernel_size=8,stride=2,padding=3),
                nn.BatchNorm2d(output_channel),
                nn.Dropout2d(0.5),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channel , output_channel , kernel_size=8,stride=2,padding=3),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True)
            )

    def decoding_conv_freq(self,input_channel , output_channel , use_dropout):
        if use_dropout:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channel , output_channel , kernel_size=(1,8),stride=(1,2),padding=(0,3)),
                nn.BatchNorm2d(output_channel),
                nn.Dropout2d(0.5),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channel , output_channel , kernel_size=(1,8),stride=(1,2),padding=(0,3)),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True)
            )

    def input_channel_size_update_encoding(self):
        self.input_channel = int(self.next_channel)
        self.next_channel = int(self.next_channel * 2)
    
    def input_channel_size_update_decoding(self):
        self.input_channel = int(self.next_channel * 2)
        self.next_channel = int(self.next_channel / 2)
