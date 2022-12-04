import torch
from torch.utils.data import DataLoader
import pathlib

import FogDatasetLoader


class DataTransforms:
    # TODO: docs
    def __init__(self, data_loader: DataLoader, window_size=512, step_size=128):
        self.input_tensor = None
        self.target_tensor = None
        self.window_size = window_size
        self.step_size = step_size
        self.loader_iter = iter(data_loader)

    def __iter__(self):
        self.position = 0
        (self.input_tensor, self.target_tensor) = next(self.loader_iter)
        self.input_tensor = self.input_tensor[0]
        self.target_tensor = self.target_tensor[0]
        self.file_length = len(self.input_tensor[0])

        self.chunky_input = None
        self.target = None
        return self

    def __next__(self):
        if self.position + self.window_size > self.file_length:
            tensors = next(self.loader_iter, None)
            self.position = 0
            if tensors:
                self.input_tensor, self.target_tensor = tensors
                self.input_tensor = self.input_tensor[0]
                self.target_tensor = self.target_tensor[0]
                self.file_length = len(self.input_tensor)
            else:
                raise StopIteration

        indices = torch.tensor([*range(self.position, self.position + self.window_size)])

        self.chunky_input = torch.index_select(self.input_tensor, dim=0, index=indices)
        self.target = torch.index_select(self.target_tensor, dim=0, index=torch.tensor([self.position + self.window_size - 1]))
        self.position += self.step_size
        return self.chunky_input, self.target

    def load_into_memory(self, batch_size) -> (torch.Tensor, torch.Tensor):
        inputs = []
        targets = []
        for idx, (input_t, target_t) in enumerate(self):
            inputs.append(input_t)
            targets.append(target_t)

        batched_inputs = []
        batched_targets = []
        for i in range(len(inputs)//batch_size):
            batched_inputs.append(torch.stack(inputs[i*batch_size: (i+1)*batch_size]))
            batched_targets.append(torch.stack(targets[i*batch_size: (i+1)*batch_size]))

        return torch.stack(batched_inputs).type(torch.float32), torch.stack(batched_targets).type(torch.float32)

    def normalize_data(self, dataset: tuple[torch.Tensor, torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        inputs = (dataset[0] - dataset[0].mean([0, 1, 2], keepdim=True))
        std = dataset[0].std([0, 1, 2], keepdim=True)
        std = torch.where(std > 0, std, 1)
        inputs = inputs / std

        return inputs, dataset[1]

    def shuffle(self, dataset: tuple[torch.Tensor, torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        rand_ind = torch.randperm(dataset[0].shape[0])
        inputs = dataset[0][rand_ind]
        targets = dataset[1][rand_ind]
        return inputs, targets


if __name__ == '__main__':
    print(pathlib.PurePath('./data/001').__str__())
    fdl = FogDatasetLoader.FogDatasetLoader('./data/001')
    loader = DataLoader(fdl, batch_size=1, shuffle=False)
    dt = DataTransforms(loader)

    inputs, targets = dt.load_into_memory(16)
    print(targets)
    print(inputs)
