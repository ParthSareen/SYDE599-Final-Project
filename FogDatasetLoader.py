import pathlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import re

'''
TODO:
    - IMU data management
    - Sliding window to produce sequences
    - EDA (outside of code)
    - Fourier Transform
'''
# Indices refer to EMG columns after removal of EEG data
swap_rules = {
    "001": [(0, 1)],
    "002": [(0, 1)],
    "003": [],
    "004": [],
    "005": [],
    "006": [(0, 1)],
    "007": [(0, 1)],
    "008-1": [(0, 1)],
    "008-2": [(1, 4), (0, 1)],
    "009": [(3, 4)],
    "010": [],
    "011": [],
    "012": [],
}


class FogDatasetLoader(Dataset):
    def __init__(self, rootdir='./data') -> None:
        super().__init__()
        self.data_paths = []

        path = pathlib.Path(rootdir)
        for file in path.rglob("*.txt"):
            print(file.as_posix())
            self.data_paths.append(file.as_posix())
        self.data_paths.sort()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        file_path = self.data_paths[index]
        df = pd.read_csv(file_path, header=None)

        # Remove EEG columns
        df.drop(df.iloc[:, 2:27], inplace=True, axis=1)
        # Remove entry number and timestamp columns
        df.drop(df.iloc[:, 0:2], inplace=True, axis=1)

        inputs = df.iloc[0:, 0:-1]
        targets = df.iloc[0:, -1]
        input_values = inputs.values
        target_values = targets.values

        self.restructure_emg(input_values, file_path)

        return torch.from_numpy(input_values), torch.from_numpy(target_values)

    def restructure_emg(self, input_values, file_path):
        # Our EMG structure: L-TA, R-TA, IO, ECG, R-GS

        # Extract experiment ID (aka patient ID) from file path
        x = re.search('\/([0-9]+)', file_path)
        experiment_id = x.group()[1:]

        # 008 contains data for 2 experiments with different swaps required for each
        if experiment_id == '008':
            if file_path.__contains__("OFF_1"):
                experiment_id = experiment_id + '-1'
            else:
                experiment_id = experiment_id + '-2'

        # Swap columns according to predefined required swaps for each experiment (from word doc)
        swap_rule = swap_rules[experiment_id]
        for swap in swap_rule:
            self.swap_columns(input_values, swap[0], swap[1])

    @staticmethod
    def swap_columns(input_values, col_index1, col_index2):
        input_values[:, [col_index2, col_index1]] = input_values[:, [col_index1, col_index2]]


if __name__ == '__main__':
    dataset = FogDatasetLoader('data')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    torch.set_printoptions(edgeitems=6)
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(batch_idx, inputs.shape, targets.shape)
