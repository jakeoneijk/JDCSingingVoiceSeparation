from Hparams import HParams
from Util import Util
import numpy as np
import pandas as pd
import pickle
import librosa
import os

class PreProcess():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.util = Util()
    
    def pre_process(self,data_path,output_path,song_list,test_song_list):
        for i,song_name in enumerate(song_list):
            print("\n---------------------\n")
            print(f'preprocess {song_name} ({i}/{len(song_list)})')

            if song_name in test_song_list:
                preprosessed_path = output_path + "/test"
            else:
                preprosessed_path = output_path + "/train"
            vocal_path = data_path + "/vocal_accom/"+song_name+"_vocal.wav"
            accom_path = data_path + "/vocal_accom/"+song_name+"_accom.wav"
            mix_path = data_path + "/mix/"+song_name+"_mix.wav"

            mag_mix,normalize_value = self.util.magnitude_spectrogram(mix_path,self.h_params.preprocess.sample_rate,
                                                    self.h_params.preprocess.nfft,
                                                    self.h_params.preprocess.window_size,
                                                    self.h_params.preprocess.hop_length)
            
            mag_vocal = self.util.magnitude_spectrogram(vocal_path,self.h_params.preprocess.sample_rate,
                                                    self.h_params.preprocess.nfft,
                                                    self.h_params.preprocess.window_size,
                                                    self.h_params.preprocess.hop_length,normalize=False)
            
            mag_accom = self.util.magnitude_spectrogram(accom_path,self.h_params.preprocess.sample_rate,
                                                    self.h_params.preprocess.nfft,
                                                    self.h_params.preprocess.window_size,
                                                    self.h_params.preprocess.hop_length,normalize=False)
            
            mag_vocal = mag_vocal/normalize_value
            mag_accom = mag_accom/normalize_value

            mag_vocal[mag_vocal>mag_mix] = mag_mix[mag_vocal>mag_mix]
            mag_accom[mag_accom>mag_mix] = mag_mix[mag_accom>mag_mix]

            num_time_frames = mag_mix.shape[1]

            if self.h_params.data.use_jdc:
                jdc_mag_mix,_ = self.util.magnitude_spectrogram(mix_path,self.h_params.preprocess.jdc_sampling_rate,
                                                            self.h_params.preprocess.jdc_nfft,
                                                            self.h_params.preprocess.jdc_window_size,
                                                            self.h_params.preprocess.jdc_hop_length)
                time_frame = librosa.core.frames_to_time(np.arange(jdc_mag_mix.shape[1]),
                                               sr=self.h_params.preprocess.jdc_sampling_rate,hop_length=self.h_params.preprocess.jdc_hop_length,n_fft=self.h_params.preprocess.jdc_nfft)
                pitch_label, is_voice_label = self.melody_labels_time_frame(data_path +"/melody1/"+song_name+"_MELODY1.csv",time_frame)
            
            for start_idx in range(0,num_time_frames,self.h_params.preprocess.model_input_time_frame_size):
                end_idx = start_idx + self.h_params.preprocess.model_input_time_frame_size
                mag_mix_seg = mag_mix[1:,start_idx:end_idx]
                mag_vocal_seg = mag_vocal[1:,start_idx:end_idx]
                mag_accom_seg = mag_accom[1:,start_idx:end_idx]

                if mag_mix_seg.shape[1] != self.h_params.preprocess.model_input_time_frame_size:
                    continue

                data = {
                    "name": song_name,
                    "mix": np.transpose(mag_mix_seg),
                    "vocal": np.transpose(mag_vocal_seg),
                    "accom": np.transpose(mag_accom_seg),
                    "start_idx":start_idx
                    }
                if self.h_params.data.use_jdc:
                    data["mix_jdc"] = np.transpose(jdc_mag_mix[:,start_idx:end_idx])
                    data["pitch_label"] = pitch_label[start_idx:end_idx]
                    data["is_voice_label"] = is_voice_label[start_idx:end_idx]
                
                save_path = os.path.join(preprosessed_path,f'{song_name}_{start_idx}.pkl')
                print(f'Saving: {save_path}')
                with open(save_path,'wb') as writing_file:
                    pickle.dump(data,writing_file)

    def melody_labels_time_frame(self, melody_path, time_frame):
        melody_sec_hz = pd.read_csv(melody_path,header=None, names=['sec','hz'])
        pitch_label = []
        is_voice_label = []
        for start_idx in range(len(time_frame)):
            start_time = time_frame[start_idx]
            if start_idx + 1 < len(time_frame):
                end_time = time_frame[start_idx+1]
                melodies_in_time = melody_sec_hz[melody_sec_hz['sec'].between(start_time, end_time)]
            else:
                melodies_in_time = melody_sec_hz[melody_sec_hz['sec'] > start_time]
        
            melodies_in_time = melodies_in_time[melodies_in_time['hz']> self.util.nonvoice_threshold]
            if len(melodies_in_time) == 0:
                label = self.util.label_nonvoice
                voice = False
            else:
                label = self.util.label_by_freq(melodies_in_time['hz'].mean())
                voice = True
        
            pitch_label.append(label)
            is_voice_label.append(1 if voice else 0)
        return np.array(pitch_label), np.array(is_voice_label)
         
            