import torch
from torch import nn
from .Trainer import Trainer,TrainState
import numpy as np
from .Model.UnetOnly import UnetOnly

class UNETTrainer(Trainer):
    def __init__(self,h_params,seed: int = None):
        super().__init__(h_params,seed)
        self.model = UnetOnly(self.h_params.resource.device).to(self.h_params.resource.device)
        self.criterias = {}
        self.criterias["loss_unet_vocal"] = nn.L1Loss(reduction='sum')
        self.criterias["loss_unet_accom"] = nn.L1Loss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.h_params.train.lr, weight_decay=1e-4)

    def run_step(self,data,metric):
        """
        run 1 step
        return loss,metric
        """
        input_unet = data['mix'].to(self.h_params.resource.device)
        input_unet = input_unet.unsqueeze(dim=1)
        vocal_target = data['vocal'].to(self.h_params.resource.device)
        vocal_target = vocal_target.unsqueeze(dim=1)
        accom_target = data['accom'].to(self.h_params.resource.device)
        accom_target = accom_target.unsqueeze(dim=1)

        
        vocal_mask,accom_maks= self.model(input_unet)

        vocal_hat = input_unet*vocal_mask
        accom_hat = input_unet*accom_maks
        vocal_loss = self.criterias["loss_unet_vocal"](vocal_hat,vocal_target)
        accom_loss = self.criterias["loss_unet_accom"](accom_hat,accom_target)

        total_loss = vocal_loss + accom_loss 

        metric["total_loss"] = np.append(metric["total_loss"],total_loss.item())
        metric["vocal_loss"] = np.append(metric["vocal_loss"],vocal_loss.item())
        metric["accom_loss"] = np.append(metric["accom_loss"],accom_loss.item())

        return total_loss,metric
        

    def metric_init(self):
        """
        return np array of chosen metric form
        """
        return {"total_loss":np.array([]),"vocal_loss":np.array([]),"accom_loss":np.array([])}

    def save_best_model(self,prev_best_metric, current_metric):
        """
        compare what is the best metric
        If current_metric is better, 
            1.save best model
            2. self.best_valid_epoch = self.current_epoch
        Return
            better metric
        """
        if prev_best_metric is None:
            return current_metric
        
        if np.mean(prev_best_metric["vocal_loss"]) > np.mean(current_metric["vocal_loss"]):
            self.save_module("vocal"+self.model.__class__.__name__,"best")
            self.best_valid_epoch = self.current_epoch
            return current_metric
        else:
            return prev_best_metric
    
    def log_metric(self, metrics ,data_size: int,train_state=TrainState.TRAIN):
        """
        log and tensorboard log
        """
        x_axis = self.global_step if train_state == TrainState.TRAIN else self.current_epoch
        log = f'Epoch ({train_state.value}): {self.current_epoch:03} ({self.local_step}/{data_size}) global_step: {self.global_step}\t'
        
        for metric_name in metrics:
            val = np.mean(metrics[metric_name])
            log += f' {metric_name}: {val:.06f}'
            self.log_writer.tensorboard_log_write(f'{train_state.value}/{metric_name}',x_axis,val)
        self.log_writer.print_and_log(log,self.global_step)