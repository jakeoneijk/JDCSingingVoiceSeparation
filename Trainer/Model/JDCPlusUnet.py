import torch
import torch.nn as nn
from .JDCNet import JDCNet
from .JDCOptions import JDCOptions
from .Unet import Unet
import math
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display

class JDCPlusUnet(nn.Module):
    def __init__(self,device):
        super(JDCPlusUnet,self).__init__()
        self.device = device
        self.jdc_options = JDCOptions()
        self.sampling_rate = 44100
        self.n_fft = 4096
        self.len_frequency_index = int(self.n_fft / 2) + 1
    
        self.jdc_block = JDCNet()
        self.w_harmonic = torch.nn.Parameter(torch.ones(301),requires_grad=True)
        self.sigma_harmonic = torch.nn.Parameter(torch.ones(301)/10,requires_grad=True)
        self.u_net = Unet(input_channel=2)
        self.final_activaion = nn.Sigmoid()
    
    def forward(self,input_jdc,input_unet):
        pitch_classification , vocal_detection = self.jdc_block(input_jdc)

        pitch_hz = self.make_pitch_array(pitch_classification)
        harmonic_structure = self.harmonic_structure_made(pitch_hz)
        input_for_unet = torch.cat([harmonic_structure[:,:,:,1:],input_unet],dim=1)
        padding = torch.zeros(input_for_unet.shape[0],input_for_unet.shape[1],1,input_for_unet.shape[3]).to(self.device)

        input_for_unet = torch.cat([input_for_unet,padding],dim=2)
        mask_vocal, mask_accom = self.u_net(input_for_unet)
        mask_vocal = self.final_activaion(mask_vocal)
        mask_accom = self.final_activaion(mask_accom)
        masked_vocal = mask_vocal[:,:,:31,:]
        masked_accom = mask_accom[:,:,:31,:]
        return pitch_classification, vocal_detection, masked_vocal,masked_accom

    def make_pitch_array(self,output_jdc):
        _, pitch_classes = output_jdc.max(dim=2)
        non_voice_idx = (pitch_classes == 0)
        pitch_midi = (self.jdc_options.min_pitch_midi + 
        ((1.0/self.jdc_options.resolution)*(pitch_classes-1)))
        pitch_hz = 2 ** ((pitch_midi - 69) / 12.) * 440
        pitch_hz[non_voice_idx] = 0
        return pitch_hz
    
    def harmonic_structure_made(self,pitch_hz):
        print("make harmonic structure")
        harmonic_structure = torch.zeros(pitch_hz.shape[0],pitch_hz.shape[1],self.len_frequency_index).to(self.device)
        pitch_hz_info = pitch_hz.clone().detach()
        pitch_hz_info[pitch_hz_info<= 0] = -1
        pitch_hz_info[pitch_hz_info > (self.sampling_rate/2)] = -1
        if len(pitch_hz_info[pitch_hz_info>0]) == 0:
            harmonic_structure = harmonic_structure.unsqueeze(dim=1)
            return harmonic_structure

        #masking
        min_masking_hz = torch.min(pitch_hz_info[pitch_hz_info>0])
        for_while_break = min_masking_hz.clone().detach()
        harmony_index = 0
        while for_while_break < (self.sampling_rate/2):
            gaussian_kernel = torch.arange(self.len_frequency_index).to(self.device)
            gaussian_kernel = torch.true_divide(gaussian_kernel,self.n_fft) * self.sampling_rate
            gaussian_kernel = gaussian_kernel.repeat(pitch_hz.shape[0]*pitch_hz.shape[1])
            gaussian_kernel = gaussian_kernel.view(pitch_hz.shape[0],pitch_hz.shape[1],-1)

            pitch_hz_for_mean = pitch_hz_info.unsqueeze(dim=2)
            gaussian_denominator = (self.sigma_harmonic[harmony_index] ** 2)/2
            if (gaussian_denominator == 0).sum() > 0:
                print("denominator is zero")
            gaussian_kernel = torch.exp((-1*((gaussian_kernel- pitch_hz_for_mean)**2))*gaussian_denominator)
            gaussian_kernel = gaussian_kernel * self.w_harmonic[harmony_index]
            gaussian_kernel[pitch_hz_info<0]=0
            harmonic_structure = harmonic_structure + gaussian_kernel
            
            pitch_hz_info[pitch_hz_info>0] = pitch_hz_info[pitch_hz_info>0] + pitch_hz[pitch_hz_info>0]
            pitch_hz_info[pitch_hz_info > (self.sampling_rate/2)] = -1
            for_while_break += min_masking_hz
            harmony_index += 1
        harmonic_structure = harmonic_structure.unsqueeze(dim=1)

        if torch.isnan(harmonic_structure).sum() > 0:
            print("there is a non in harmonicstructure")
        return harmonic_structure
    
    def display_torch(self,tensor_to_display):
        tensor_to_display_numpy = tensor_to_display.detach().numpy()
        tensor_to_display_numpy= tensor_to_display_numpy.reshape(int(tensor_to_display.shape[0]*tensor_to_display.shape[1]),-1)
        S = librosa.amplitude_to_db(abs(tensor_to_display_numpy))
        plt.figure(figsize=(15,5))
        librosa.display.specshow(S, sr=self.sampling_rate, hop_length=1024, x_axis='time', y_axis='linear')
        
if __name__ == '__main__':
    print("jdc plus Unet model size test")
    dummy = torch.randn((10,1,31,513))
    dummy_2 = torch.rand((10,1,31,2048))
    jdc_plus_unet_model = JDCPlusUnet()
    output1,output2,output3,output4 = jdc_plus_unet_model(dummy,dummy_2)
    print(output1.size())
    print(output2.size())
    print(output3.size())
    print(output4.size())

