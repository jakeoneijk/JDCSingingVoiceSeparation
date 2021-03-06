import argparse
import os
import torch
from dataclasses import dataclass
from datetime import datetime
time_for_output = datetime.now().strftime('%y%m%d-%H%M%S') + "_"

class HParams(object):
    def __init__(self):

        self.mode = Mode()
        self.resource = Resource()
        self.data = Data()
        self.preprocess = PreProcess()
        self.train= Train()
        self.log = Logging()
        self.test = Test()
        self.evaluate = Evaluate()

        self.make_essential_dir()

    def make_essential_dir(self):
        os.makedirs(self.data.root_path,exist_ok=True)

        for data_name in self.data.name_list:
            data_path = os.path.join(self.data.root_path,data_name)
            os.makedirs(data_path,exist_ok=True)
            os.makedirs(data_path+"/Preprocessed",exist_ok=True)
            os.makedirs(data_path+"/Preprocessed"+"/train",exist_ok=True)
            os.makedirs(data_path+"/Preprocessed"+"/test",exist_ok=True)

        if self.mode.app == "train":
            os.makedirs(self.log.log_root_path,exist_ok=True)
            os.makedirs(self.log.log_path,exist_ok=True)
            os.makedirs(self.log.tensorboard_path,exist_ok=True)
            os.makedirs(self.log.model_save_path,exist_ok=True)
        
        os.makedirs(self.test.pretrain_path,exist_ok=True)
        os.makedirs(self.test.output_path,exist_ok=True)

@dataclass
class Mode:
    app = ["make_data" , "preprocess" , "train" , "test","evaluate"][-2]
    train = ["start","resume"][0]
    debug_mode = False

@dataclass
class Resource:
    num_workers = 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@dataclass
class Data:
    valid_ratio = 0.1
    original_data_path = "../210101_data"
    root_path = "./Data"
    name_list = ["MedlyDBSVS"] #"MedlyDBSVS"
    use_jdc = True

@dataclass
class PreProcess:
    sample_rate = 44100
    nfft = 4096
    window_size = 3528
    hop_length = 441
    jdc_sampling_rate = 8000
    jdc_nfft =1024
    jdc_window_size=1024
    jdc_hop_length=80
    model_input_time_frame_size = 31

@dataclass
class Train:
    model = ["JDCUNET","UNETONLY"][1]
    batch_size = 16
    lr = 0.001
    epoch = 300

@dataclass
class Logging():
    log_root_path = "./Log"
    log_path = os.path.join(log_root_path,time_for_output)
    log_name = os.path.join(log_path,"log.txt")
    tensorboard_path = os.path.join(log_path,"tb")
    model_save_path = log_path
    model_save_name = ""
    log_every_local_step = 200
    time = time_for_output

@dataclass
class Test():
    input_path = "./TestInput/testset/"
    test_type = "test_set"
    audio_file = ["AimeeNorwich_Child_mix.wav","AvaLuna_Waterduct_mix.wav","FamilyBand_Again_mix.wav","MusicDelta_Country2_mix.wav","MusicDelta_Gospel_mix.wav"] #"I'm in love(AR).wav"
    output_path = "./TestOutput"
    pretrain_path = "./Pretrained/"
    pretrain_name = "UnetOnly.pth"
    is_binary_mask = True

@dataclass
class Evaluate():
    input_path = "./EvaluateInput"
    model_name = "JDCUnet" #"JDCUnet" #"UnetOnly"