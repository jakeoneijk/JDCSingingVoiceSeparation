import os
import torch.utils.data.dataset as dataset
import pickle

class DataSet(dataset.Dataset):

    def __init__(self, data_path):
        self.files = []
        for fname in data_path:
            self.files.append(self.read_data(fname))
    
    def read_data(self, data_path):
        with open(data_path, 'rb') as pickle_file:
            file_data_dict = pickle.load(pickle_file)
        return file_data_dict
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return self.files[index]