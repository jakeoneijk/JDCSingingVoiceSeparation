import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from PreProcess import PreProcess
from Hparams import HParams
from Trainer.Model.JDCPlusUnet import JDCPlusUnet
from Trainer.Model.UnetOnly import UnetOnly
from Util import Util

class Test():
    def __init__(self,h_params:HParams):
        self.pre_processor = PreProcess(h_params)
        self.device = h_params.resource.device
        self.h_params = h_params
        self.util = Util()
        self.batch_size = self.h_params.train.batch_size

        if h_params.train.model == "JDCUNET":
            self.model = JDCPlusUnet(self.h_params.resource.device).to(self.h_params.resource.device)
        if h_params.train.model == "UNETONLY":
            self.model = UnetOnly(self.h_params.resource.device).to(self.h_params.resource.device)

        self.phase = None
        self.normalize_value = 0
        self.output_path = None
        self.output_name = ""
    

    
    def output(self,pretrain_path:str,audio_input_path,audio_name):
        print("load pretrained model")

        pretrain_name = pretrain_path.split("/")[-1].replace(".pth","")+ "/"
        self.output_path = self.h_params.test.output_path + "/" + pretrain_name
        os.makedirs(self.output_path,exist_ok=True)

        self.output_name = self.h_params.log.time + audio_name.replace(".wav","")

        best_model_load = torch.load(pretrain_path,map_location='cpu')
        self.model.load_state_dict(best_model_load)
        self.model.to(self.device)

        #self.input_to_output(audio_input_path)
        if self.h_params.train.model == "JDCUNET":
            input_jdc,input_unet = self.make_model_input_jdcunet(audio_input_path)
        if self.h_params.train.model == "UNETONLY":
            input_unet = self.make_model_input(audio_input_path)

        mask_vocal=None
        mask_accom = None
        just_test_input = None

        for start_idx in range(0,input_unet.shape[0],self.batch_size):
            print(f"making mask {start_idx}/{input_unet.shape[0]}")
            if self.h_params.train.model == "JDCUNET":
                input_seg = input_unet[start_idx:start_idx+self.batch_size]
                input_seg = input_seg.to(self.device)
                input_jdc_seg = input_jdc[start_idx:start_idx+self.batch_size]
                input_jdc_seg = input_jdc_seg.to(self.device)
                with torch.no_grad():
                    mask = self.make_mask_JDCUNET(input_jdc_seg,input_seg)

            if self.h_params.train.model == "UNETONLY":
                input_seg = input_unet[start_idx:start_idx+self.batch_size]
                input_seg = input_seg.to(self.device)
                with torch.no_grad():
                    mask = self.make_mask_UNET(input_seg)

            if self.h_params.test.is_binary_mask:
                mask[mask>0.5] = 1
                mask[mask<=0.5] = 0

            mask_a = torch.ones_like(mask) - mask
            just_test_input = self.make_output_spectro(just_test_input,input_seg)  
            mask_vocal = self.make_output_spectro(mask_vocal,mask)
            mask_accom = self.make_output_spectro(mask_accom,mask_a)

        just_test_input_numpy = self.torch_to_np_spec(just_test_input)
        mask_v_numpy = self.torch_to_np_spec(mask_vocal)
        mask_a_numpy = self.torch_to_np_spec(mask_accom)
        self.mask_histogram(mask_v_numpy)
        masked_v_numpy = just_test_input_numpy * mask_v_numpy * self.normalize_value
        masked_a_numpy = just_test_input_numpy * mask_a_numpy * self.normalize_value
        just_test_input_numpy = just_test_input_numpy * self.normalize_value

        self.plot_spec(masked_v_numpy,"vocal")
        #self.plot_spec(just_test_input_numpy,"input")
        #self.inverse_stft_griffin_lim(masked_v_numpy,"vocal")
        #self.inverse_stft_griffin_lim(masked_a_numpy,"accom")
        self.inverse_stft(masked_v_numpy*self.phase[:,:masked_v_numpy.shape[1]],"vocal")
        self.inverse_stft(masked_a_numpy*self.phase[:,:masked_a_numpy.shape[1]],"accom")
        self.inverse_stft(just_test_input_numpy*self.phase[:,:masked_v_numpy.shape[1]],"restored_input") 
    
    def make_output_spectro(self,masking_spectro,output_model):
        result = masking_spectro
        output_model = output_model.to(torch.device('cpu'))
        for i in range(0,output_model.shape[0]):
            output_seg = (output_model[i].squeeze(0)).transpose(0,1)
            result = self.concatenation(result,output_seg)
        return result
    
    def concatenation(self,x,y,dimension=1):
        if x is None:
            return y
        else:
            return torch.cat((x,y),1)
    
    def make_mask_JDCUNET(self,jdc_seg,unet_seg):
         pitch,voice,vocal_mask,accom_maks = self.model(jdc_seg,unet_seg)
         return vocal_mask
    
    def make_mask_UNET(self,unet_seg):
        vocal_mask,accom_maks = self.model(unet_seg)
        return vocal_mask
    
    def make_model_input(self,audio_input_path):
        mag_unet,normalize_value,phase = self.util.magnitude_spectrogram(audio_input_path,self.h_params.preprocess.sample_rate,
                                                    self.h_params.preprocess.nfft,
                                                    self.h_params.preprocess.window_size,
                                                    self.h_params.preprocess.hop_length,get_pahse=True)
        self.phase = phase
        self.normalize_value = normalize_value

        input_unet = []
        for start_idx in range(0,mag_unet.shape[1],self.h_params.preprocess.model_input_time_frame_size):
            end_idc = start_idx + self.h_params.preprocess.model_input_time_frame_size
            input_unet_seg = np.swapaxes(mag_unet[1:,start_idx:end_idc],axis1=0,axis2=1)
            if input_unet_seg.shape[0] != self.h_params.preprocess.model_input_time_frame_size:
                continue
            input_unet.append([input_unet_seg])
          
        input_unet_tensor = torch.tensor(input_unet)

        return input_unet_tensor
    
    def make_model_input_jdcunet(self,audio_input_path):
        mag_unet,normalize_value,phase = self.util.magnitude_spectrogram(audio_input_path,self.h_params.preprocess.sample_rate,
                                                    self.h_params.preprocess.nfft,
                                                    self.h_params.preprocess.window_size,
                                                    self.h_params.preprocess.hop_length,get_pahse=True)
        self.phase = phase
        self.normalize_value = normalize_value

        jdc_mag_mix,_ = self.util.magnitude_spectrogram(audio_input_path,self.h_params.preprocess.jdc_sampling_rate,
                                                            self.h_params.preprocess.jdc_nfft,
                                                            self.h_params.preprocess.jdc_window_size,
                                                            self.h_params.preprocess.jdc_hop_length)
        input_unet = []
        input_jdc = []

        for start_idx in range(0,mag_unet.shape[1],self.h_params.preprocess.model_input_time_frame_size):
            end_idx = start_idx + self.h_params.preprocess.model_input_time_frame_size
            input_unet_seg = np.transpose(mag_unet[1:,start_idx:end_idx])
            input_jdc_seg = np.transpose(jdc_mag_mix[:,start_idx:end_idx])

            if input_unet_seg.shape[0] != self.h_params.preprocess.model_input_time_frame_size:
                continue

            input_unet.append([input_unet_seg])
            input_jdc.append([input_jdc_seg])
          
        input_unet_tensor = torch.tensor(input_unet)
        input_jdc_tensor = torch.tensor(input_jdc)

        return input_jdc_tensor,input_unet_tensor
    
    def inverse_stft_griffin_lim(self,stft_mat,name):
        filename = self.output_path +self.h_params.time_for_output+"_griffin_" +self.output_name+"_"+name+".wav"
        istft_mat = librosa.griffinlim(abs(stft_mat), hop_length=self.h_params.stft_hop_length, win_length=self.h_params.stft_window_size)
        sf.write(filename, istft_mat, self.h_params.down_sample_rate)
        return istft_mat
    
    def inverse_stft(self,stft_mat,name):
        filename = self.output_path +self.output_name+"_"+name+".wav"
        istft_mat = librosa.core.istft(stft_mat, hop_length=self.h_params.preprocess.hop_length,  win_length=self.h_params.preprocess.window_size)
        sf.write(filename, istft_mat, self.h_params.preprocess.sample_rate)
        return istft_mat
    
    def plot_spec(self,spectro,name):
        filename = self.output_path +self.output_name+"_"+name+".png"
        plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(spectro), ref=np.max))
        plt.colorbar()
        plt.savefig(filename, dpi=600)
    
    def mask_histogram(self,mask,name="vocal_mask_histogram"):
        plt.hist(mask.flatten())
        plt.savefig(self.output_path+"_"+name+".png", dpi=300)
    
    def torch_to_np_spec(self,torch_spec):
        numpy_spec = torch_spec.detach().cpu().numpy()
        numpy_spec = numpy_spec
        zero_padd = np.zeros((1,numpy_spec.shape[1]))
        numpy_spec= np.concatenate((zero_padd,numpy_spec),0)
        return numpy_spec
    
    def input_to_output(self,audio_input_path):
        y = self.pre_processor.audio_load(audio_input_path,self.h_params.test_audio_sample_rate)
        filename = self.output_path+"_original_input.wav"
        sf.write(filename, y, self.h_params.down_sample_rate)


'''

class Test():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.model = None
        self.util = Util()
    
    def run(self,pretrain_path,audio_input_file,model_name):
        if model_name == "JDCUNET":
            self.model = JDCPlusUnet(self.h_params.resource.device).to(self.h_params.resource.device)
            mag_mix,normalize_value,jdc_mag_mix,phase = self.make_model_input_jdcunet(self,self.h_params.test.input_path + audio_input_file)

        model_load = torch.load(pretrain_path,map_location='cpu')
        self.model.load_state_dict(model_load)
        self.model.to(self.device)

        num_time_frames = mag_mix.shape[1]

        for start_idx in range(0,num_time_frames,self.h_params.preprocess.model_input_time_frame_size):
            end_idx = start_idx + self.h_params.preprocess.model_input_time_frame_size

            if end_idx >= num_time_frames:
                break

            if model_name == "JDCUNET":
                self.make_mask_JDCUNET(np.transpose(mag_mix[1:,start_idx:end_idx]),np.transpose(jdc_mag_mix[:,start_idx:end_idx]))
    
    def make_mask_JDCUNET(self,mag_mix,jdc_mag_mix):



    
    '''

