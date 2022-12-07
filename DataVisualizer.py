import torch
import numpy as np
import matplotlib.pyplot as plt

from DataTransforms import DataLoader, DataTransforms
from FogDatasetLoader import FogDatasetLoader


def plot_all_data(loader, column_order):
    for inputs, targets in loader:
        inputs = torch.squeeze(inputs)
        targets = torch.squeeze(targets).unsqueeze(-1)
        data = torch.cat([inputs, targets], dim=1)
        print(inputs.shape, targets.shape, data.shape)
        for i in range(inputs.shape[-1]):
            plt.figure(i)
            plt.scatter(range(inputs.shape[0]), inputs[:, i], s=0.01)

    for i in range(34):
        plt.figure(i)
        plt.title(column_order[i])
        plt.savefig(f"visualized_data/{i}.png")


if __name__ == "__main__":
    fdl = FogDatasetLoader('./data')
    loader = DataLoader(fdl, batch_size=1, shuffle=False)

    # EMG structure: L-TA, R-TA, IO, ECG, R-GS
    signals_order = ["L-TA", "R-TA", "IO", "ECG", "R-GS"]
    imu_order = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z", "NC/SC"]
    imu_positions = ["left shank", "right shank", "waist", "arm"]
    imu_order = [f"{pos}: {d_type}" for pos in imu_positions for d_type in imu_order]
    column_order = signals_order + imu_order + ["targets"]

    maximums = []
    minimums = []
    means = []
    std_devs = []
    for inputs, targets in loader:
        inputs = torch.squeeze(inputs)
        targets = torch.squeeze(targets).unsqueeze(-1)
        data = torch.cat([inputs, targets], dim=1)
        print(inputs.shape, targets.shape, data.shape)

        maximums.append(data.max(0).values.numpy())
        minimums.append(data.min(0).values.numpy())
        means.append(data.mean(0).numpy())
        std_devs.append(data.std(0).numpy())

    maximums = np.array(maximums)
    minimums = np.array(minimums)
    means = np.array(means)
    std_devs = np.array(std_devs)

    for i in range(34):
        plt.figure(i)
        plt.scatter(range(maximums.shape[0]), maximums[:, i])
        plt.scatter(range(minimums.shape[0]), minimums[:, i])
        plt.scatter(range(means.shape[0]), means[:, i])
        plt.scatter(range(std_devs.shape[0]), std_devs[:, i])
        plt.legend(["max", "min", "mean", "std dev"])
        plt.title(column_order[i])
        plt.savefig(f"visualized_data/metadata/{i}.png")
        plt.close(i)
