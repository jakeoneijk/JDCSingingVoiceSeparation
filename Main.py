import os
import numpy as np
import random
from Hparams import HParams
import math
from DataSet import DataSet
from torch.utils.data import DataLoader, Dataset
from MakeData import MakeData
from PreProcess import PreProcess
from Trainer.JDCUNETTrainer import JDCUNETTrainer
from Trainer.UNETTrainer import UNETTrainer
from Test import Test
from Evaluation import Evaluation

class AppController():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
    
    def run(self):
        if self.h_params.mode.app == "make_data":
            self.make_data()

        if self.h_params.mode.app == "preprocess":
            self.preprocess()
        
        if self.h_params.mode.app == "train":
            self.train()
        
        if self.h_params.mode.app == "test":
            self.test()
        
        if self.h_params.mode.app == "evaluate":
            self.evaluate()
        
        print("finish app")

    def make_data(self):
        make_data = MakeData()
        for data_name in self.h_params.data.name_list:
            data_path = self.h_params.data.original_data_path+"/"+data_name
            song_list = make_data.get_songlist_from_vocal_accom_path(data_path+"/vocal_accom")
            make_data.make_mix(song_list,data_path+"/vocal_accom",data_path+"/mix")

    def preprocess(self):
        preprocessor = PreProcess(self.h_params)
        for data_name in self.h_params.data.name_list:
            data_output_root = os.path.join(self.h_params.data.root_path,data_name) + "/Preprocessed"
            data_root = self.h_params.data.original_data_path + "/"+data_name
            test_song_list = np.genfromtxt(data_root+"/test_song.txt",dtype="str")
            song_list = [fname.split("_mix")[0] for fname in os.listdir(data_root+"/mix")]
            preprocessor.pre_process(data_root,data_output_root,song_list,test_song_list)

    def train(self):
        total_train_path_list = []
        total_valid_path_list = []
        total_test_path_list = []
        for data_name in self.h_params.data.name_list:
            data_path = os.path.join(self.h_params.data.root_path,data_name+"/Preprocessed") 
            train_path_list,valid_path_list,test_path_list = self.get_data_path(data_path)
            total_train_path_list = total_train_path_list + train_path_list
            total_valid_path_list = total_valid_path_list + valid_path_list
            total_test_path_list = total_test_path_list + test_path_list
        
        if self.h_params.mode.debug_mode:
            total_train_path_list = total_train_path_list[:20]
            total_valid_path_list = total_valid_path_list[:20]
            total_test_path_list = total_test_path_list[:20]

        train_data_loader,valid_data_loader,test_data_loader = self.get_data_loader(total_train_path_list,total_valid_path_list,total_test_path_list)
        """
        construct trainer hete and fit
        """
        if self.h_params.train.model == "JDCUNET":
            trainer = JDCUNETTrainer(self.h_params)
        elif self.h_params.train.model == "UNETONLY":
            trainer = UNETTrainer(self.h_params)
        trainer.set_data_loader(train_data_loader,valid_data_loader,test_data_loader)
        trainer.fit()

    def get_data_path(self,data_path):
        total_train_file_list = [os.path.join(data_path+"/train",fname) for fname in os.listdir(data_path+"/train")]
        test_file_list = [os.path.join(data_path+"/test",fname) for fname in os.listdir(data_path+"/test")]
        #divide train and valid
        num_total_train_data = len(total_train_file_list)
        total_indices = list(range(num_total_train_data))
        random.shuffle(total_indices)
        num_train_set = math.floor(num_total_train_data * (1-self.h_params.data.valid_ratio))
        train_idx = total_indices[:num_train_set]
        valid_idx = total_indices[num_train_set:]

        train_file_list = [total_train_file_list[i] for i in train_idx]
        valid_file_list = [total_train_file_list[i] for i in valid_idx]

        return train_file_list,valid_file_list,test_file_list

    def get_data_loader(self,train_path_list,valid_path_list,test_path_list):
        
        train_data_set = DataSet(train_path_list)
        valid_data_set = DataSet(valid_path_list)
        test_data_set = DataSet(test_path_list)

        train_data_loader = DataLoader(train_data_set,pin_memory=True,batch_size=self.h_params.train.batch_size, 
        shuffle=True, num_workers=self.h_params.resource.num_workers, drop_last=True)
        valid_data_loader = DataLoader(valid_data_set,pin_memory=True,batch_size=self.h_params.train.batch_size, 
        shuffle=False, num_workers=self.h_params.resource.num_workers, drop_last=True)
        test_data_loader = DataLoader(test_data_set, batch_size=self.h_params.train.batch_size, 
        shuffle=False, num_workers=self.h_params.resource.num_workers, drop_last=False)

        return train_data_loader,valid_data_loader,test_data_loader



    def test(self):
        tester = Test(self.h_params) 
        if self.h_params.test.test_type == "test_set":
            audio_list = os.listdir(self.h_params.test.input_path)
            for i,test_audio in enumerate(audio_list):
                print(f'{i+1}/{len(audio_list)}')
                tester.output(pretrain_path=self.h_params.test.pretrain_path+self.h_params.test.pretrain_name,
                                audio_input_path=self.h_params.test.input_path+test_audio,
                                audio_name=f"{test_audio}")
        else:
            tester.output(pretrain_path=self.h_params.test.pretrain_path+self.h_params.test.pretrain_name,
                            audio_input_path=self.h_params.test.input_path+self.h_params.test.audio_file,
                            audio_name=f"{self.h_params.test.audio_file}")
    
    def evaluate(self):
        eval = Evaluation(self.h_params)
        eval.evaluate_test_set()

if __name__ == '__main__':
    h_params = HParams()
    app_controller = AppController(h_params)
    app_controller.run()