import torch
import pandas as pd
import numpy as np


ETTh1_CHANNELS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


class ETTh1DataLoader:

    def __init__(self, csv_path, n_samples=1000, seq_len=96, batch_size=32,
                 target_column='OT', start_idx=0, single_sequence=False,
                 use_all_channels=False):
        self.csv_path = csv_path
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.target_column = target_column
        self.start_idx = start_idx
        self.single_sequence = single_sequence
        self.use_all_channels = use_all_channels
        self.data = None

    def load_data(self):
        df = pd.read_csv(self.csv_path)

        if self.use_all_channels:
            columns = [c for c in ETTh1_CHANNELS if c in df.columns]
            if not columns:
                raise ValueError(
                    f"No ETTh1 channels found in CSV. Expected one or more of: {ETTh1_CHANNELS}. "
                    f"Got columns: {list(df.columns)}. "
                    "Please replace the CSV with the full 7-channel ETTh1 dataset."
                )
            values = df[columns].values  # (n_rows, n_channels)
        else:
            values = df[self.target_column].values  # (n_rows,)

        if self.single_sequence:
            end = self.start_idx + self.n_samples
            if end > len(values):
                end = len(values)
            sequence = values[self.start_idx:end]

            if self.use_all_channels:
                # sequence: (n_samples, n_channels) → (1, n_samples, n_channels)
                self.data = torch.FloatTensor(sequence).unsqueeze(0)
            else:
                # sequence: (n_samples,) → (1, n_samples, 1)
                self.data = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)
        else:
            data_list = []
            for i in range(self.n_samples):
                start = self.start_idx + i * (self.seq_len // 2)
                end = start + self.seq_len

                if end > len(values):
                    break

                data_list.append(values[start:end])

            data_array = np.array(data_list)  # (n_windows, seq_len) or (n_windows, seq_len, n_channels)

            if self.use_all_channels:
                self.data = torch.FloatTensor(data_array)  # (n_windows, seq_len, n_channels)
            else:
                self.data = torch.FloatTensor(data_array).unsqueeze(-1)  # (n_windows, seq_len, 1)

        return self.data

    def get_batches(self):
        batches = []
        for i in range(0, len(self.data), self.batch_size):
            batches.append(self.data[i:i+self.batch_size])
        return batches
