import torch
import pandas as pd
import numpy as np


class ETTh1DataLoader:
    
    def __init__(self, csv_path, n_samples=1000, seq_len=96, batch_size=32, 
                 target_column='OT', start_idx=0, single_sequence=False):
        self.csv_path = csv_path
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.target_column = target_column
        self.start_idx = start_idx
        self.single_sequence = single_sequence
        self.data = None
    
    def load_data(self):
        df = pd.read_csv(self.csv_path)
        values = df[self.target_column].values
        
        if self.single_sequence:
            end = self.start_idx + self.n_samples
            if end > len(values):
                end = len(values)
            sequence = values[self.start_idx:end]
            self.data = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)
        else:
            max_start = min(len(values) - self.seq_len, self.start_idx + self.n_samples * self.seq_len)
            end_idx = self.start_idx + self.n_samples * self.seq_len
            end_idx = min(end_idx, max_start)
            
            data_list = []
            for i in range(self.n_samples):
                start = self.start_idx + i * (self.seq_len // 2)
                end = start + self.seq_len
                
                if end > len(values):
                    break
                
                sequence = values[start:end]
                data_list.append(sequence)
            
            data_array = np.array(data_list)
            self.data = torch.FloatTensor(data_array).unsqueeze(-1)
        
        return self.data
    
    def get_batches(self):
        batches = []
        for i in range(0, len(self.data), self.batch_size):
            batches.append(self.data[i:i+self.batch_size])
        return batches
