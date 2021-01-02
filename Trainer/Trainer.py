import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from LogWriter import LogWriter
from Hparams import HParams
from abc import ABC, abstractmethod
from enum import Enum,unique
import torch
from torch.utils.data import DataLoader

@unique
class TrainState(Enum):
    TRAIN = "train"
    VALIDATE = "valid"
    TEST = "test"
 
class Trainer(ABC):
    def __init__(self,h_params:HParams,seed: int = None):
        self.h_params = h_params
        self.model = None
        self.train_data_loader = None
        self.valid_data_loader = None
        self.test_data_loader = None
        self.criteria = None
        self.optimizer = None
        
        if seed is None:
            self.seed = torch.cuda.initial_seed()
            torch.manual_seed(self.seed)
        else:
            self.seed = seed
            torch.manual_seed(self.seed)

        self.check_point_num = 0 #binary
        self.current_epoch = 0
        self.total_epoch = self.h_params.train.epoch

        self.best_valid_metric = None
        self.best_valid_epoch = 0

        self.global_step = 0
        self.local_step = 0

        self.log_writer = LogWriter(self.h_params)
    
    def set_data_loader(self,train,valid,test):
        self.train_data_loader = train
        self.valid_data_loader = valid
        self.test_data_loader = test
    
    def fit(self,use_val_metric=True):
        
        for _ in range(self.current_epoch, self.total_epoch):
            self.log_writer.print_and_log(f'----------------------- Start epoch : {self.current_epoch} / {self.h_params.train.epoch} -----------------------',self.global_step)
            self.log_writer.print_and_log(f'current best epoch: {self.best_valid_epoch}',self.global_step)
            self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------',self.global_step)
    
            #Train
            self.log_writer.print_and_log('train_start',self.global_step)
            train_metric = self.run_epoch(self.train_data_loader,TrainState.TRAIN)
            
            #Valid
            self.log_writer.print_and_log('valid_start',self.global_step)
            with torch.no_grad():
                valid_metric = self.run_epoch(self.valid_data_loader,TrainState.VALIDATE)
            
            self.best_valid_metric = self.save_best_model(self.best_valid_metric, valid_metric)
            
            self.current_epoch += 1

        #Test   
        self.log_writer.print_and_log(f'test_best_epoch: {self.best_valid_epoch}',self.global_step)
        self.load_module()
        with torch.no_grad():
            test_metric = self.run_epoch(self.test_data_loader,TrainState.TEST)

        self.final_report(test_metric)
        print("Training complete")
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState):
        if train_state == TrainState.TRAIN:
            self.model.train()
        else:
            self.model.eval()
        dataset_size = len(dataloader)
        metric = self.metric_init()

        for step,data in enumerate(dataloader):
            self.local_step = step
            loss,metric = self.run_step(data,metric)
        
            if train_state == TrainState.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.local_step % self.h_params.log.log_every_local_step == 0:
                    self.log_metric(metrics=metric,data_size=dataset_size)
                self.global_step += 1
        
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST:
            self.log_metric(metrics=metric,data_size=dataset_size,train_state=train_state)

        if train_state == TrainState.TRAIN:
            self.save_checkpoint(self.check_point_num)
            self.check_point_num = int((self.check_point_num+1)%2)

        return metric

    def save_module(self,name,prefix=''):
        path = os.path.join(self.h_params.log.model_save_path,f'{prefix}_{name}.pth')
        torch.save(self.model.state_dict(), path)
    
    def load_module(self,name,prefix=''):
        path = os.path.join(self.h_params.log.model_save_path,f'{prefix}_{name}.pth')
        best_model_load = torch.load(path)
        self.model.load_state_dict(best_model_load)
    
    def save_checkpoint(self,prefix=""):
        train_state = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'seed': self.seed,
            'models': self.model.state_dict(),
            'optimizers': self.optimizer.state_dict(),
            'best_metric': self.best_valid_metric,
            'best_model_epoch' :  self.best_valid_epoch,
        }
        path = os.path.join(self.h_params.log.model_save_path,f'{self.model.__class__.__name__}_checkpoint{prefix}.pth')
        torch.save(train_state,path)

    def resume(self,filename:str):
        cpt = torch.load(filename)
        self.seed = cpt['seed']
        torch.manual_seed(self.seed)
        self.current_epoch = cpt['epoch']
        self.global_step = cpt['step']
        self.model.load_state_dict(cpt['models'])
        self.optimizer.load_state_dict(cpt['optimizers'])
        self.best_valid_result = cpt['best_metric']
        self.best_valid_epoch = cpt['best_model_epoch']
    
    @abstractmethod
    def run_step(self,data,metric):
        """
        run 1 step
        return loss,metric
        """
        raise NotImplementedError

    @abstractmethod
    def metric_init(self):
        """
        return np array of chosen metric form
        """
        raise NotImplementedError

    @abstractmethod
    def save_best_model(self,prev_best_metric, current_metric):
        """
        compare what is the best metric
        If current_metric is better, 
            1.save best model
            2. self.best_valid_epoch = self.current_epoch
        Return
            better metric
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_metric(self, metrics ,data_size: int,train_state=TrainState.TRAIN):
        """
        log and tensorboard log
        """
        raise NotImplementedError