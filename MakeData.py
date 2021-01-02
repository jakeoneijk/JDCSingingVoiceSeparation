import librosa
import os
import soundfile as sf

class MakeData:
    def get_songlist_from_vocal_accom_path(self,vocal_accom_path):
        songlist = []
        for file_name in os.listdir(vocal_accom_path):
            if "vocal" in file_name:
                songlist.append(file_name.split("_vocal")[0])
        return songlist
        
    def make_mix(self,song_list,vocal_accom_path,mix_path):
        for song_name in song_list:
            vocal,_= librosa.load(vocal_accom_path+"/"+song_name+"_vocal.wav",sr=44100,mono=True)
            accom,_= librosa.load(vocal_accom_path+"/"+song_name+"_accom.wav",sr=44100,mono=True)
            mix = vocal + accom
            sf.write(mix_path+"/"+song_name+"_mix.wav",mix,44100)

