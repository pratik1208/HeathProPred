import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        layers = [518, 256, 128, 64, 32, 16, 8, 4, 2, 1]

        # Define a list of linear layers with batch normalization
        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1]),
                    nn.BatchNorm1d(layers[i + 1]),
                    nn.LeakyReLU(0.15),
                )
                for i in range(len(layers) - 1)
            ]
        )

        # Final sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Iterate through linear layers
        for layer in self.linear_layers:
            x = layer(x)

        # Apply sigmoid activation
        x = self.sigmoid(x)

        return x


class MyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features.iloc[index, :].values, self.targets.iloc[index]
