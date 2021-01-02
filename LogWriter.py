import os
from Hparams import HParams
from torch.utils.tensorboard import SummaryWriter

class LogWriter():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.log_name = self.h_params.log.log_path + "/log.txt"
        self.tensorboard_writer = SummaryWriter(log_dir=self.h_params.log.tensorboard_path)

    def print_and_log(self,log_message,global_step):
        print(log_message)
        self.log_write(log_message,global_step)

    def log_write(self,log_message,global_step):
        if global_step == 0:
            file = open(self.log_name,'w')
            file.write("========================================="+'\n')
            file.write("Epoch :" + str(self.h_params.train.epoch)+'\n')
            file.write("lr :" + str(self.h_params.train.lr)+'\n')
            file.write("Batch :" + str(self.h_params.train.batch_size)+'\n')
            file.write("========================================="+'\n')
            file.close()

        file = open(self.log_name,'a')
        file.write(log_message+'\n')
        file.close()
    
    def tensorboard_log_write(self,name,x_axis,value):
        self.tensorboard_writer.add_scalar(name,value,x_axis)