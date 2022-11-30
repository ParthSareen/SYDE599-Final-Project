import torch
from torch.utils.data import DataLoader

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
        self.target = torch.index_select(self.target_tensor, dim=0, index=torch.tensor([self.position + self.window_size - 1])).item()
        self.position += self.step_size
        return self.chunky_input, self.target


if __name__ == '__main__':
    fdl = FogDatasetLoader.FogDatasetLoader('./data')
    loader = DataLoader(fdl, batch_size=1, shuffle=False)
    dt = DataTransforms(loader)
    for idx, (input_t, target_t) in enumerate(dt):
        print(input_t, target_t)
