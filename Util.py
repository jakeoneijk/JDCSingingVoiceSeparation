import librosa
import numpy as np

class Util():
    def __init__(self,low_midi=38, high_midi=83, num_pitch_labels=721, nonvoice_threshold=0.1):
        self.num_pitch_labels = num_pitch_labels
        self.num_labels = self.num_pitch_labels + 1
        self.label_nonvoice = self.num_labels - 1
        self.nonvoice_threshold = nonvoice_threshold

        self.label_midis = np.linspace(low_midi, high_midi, num=num_pitch_labels)
        self.label_hz = self.midi_to_frequency(self.label_midis)
        self.input_time_frame_size = 31

    def midi_to_frequency(self,midi):
        midi_A4 = 69
        hz_A4 = 440
        x = np.exp(np.log(2) / 12)
        return hz_A4 * (x ** (midi - midi_A4))

    def magnitude_spectrogram(self,audio_path,target_sr=8000,n_fft=1024,
                                win_length=1024,hop_length=80,normalize=True):
        y,sr = librosa.load(audio_path,sr=44100)
        if sr != target_sr:
            y_resample = librosa.resample(y,orig_sr=sr, target_sr=target_sr)
        else:
            y_resample = y
        spec = librosa.stft(y_resample,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        mag_spec = np.abs(spec)

        if normalize == False:
            return mag_spec

        normalize_value = np.max(mag_spec)
        normalized_mag = mag_spec/normalize_value #librosa.amplitude_to_db(mag_spec,ref=normalize_value)
        
        return normalized_mag,normalize_value
    
    def label_by_freq(self, freq):
        if freq < self.nonvoice_threshold:
            return self.label_nonvoice
        else:
            return self.interval_index(freq)
    
    def interval_index(self,freq):
        find_index = self.num_pitch_labels // 2
        start_index = 0
        end_index = self.num_pitch_labels-1
        while True:
            if self.label_hz[find_index] < freq:
                if find_index == end_index:
                    return find_index
                elif freq < self.label_hz[find_index+1]:
                    return find_index
                else:
                    start_index = find_index + 1
                    find_index = (start_index + end_index)//2
            else:
                if find_index == start_index:
                    return start_index
                elif freq > self.label_hz[find_index - 1]:
                    return find_index
                else:
                    end_index = find_index - 1
                    find_index = (start_index + end_index)//2
    
