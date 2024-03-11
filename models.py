import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll = nn.Linear(518, 256)
        self.ll2 = nn.Linear(256, 256)
        self.ll21 = nn.Linear(256, 256)
        self.ll22 = nn.Linear(256, 128)
        self.ll3 = nn.Linear(128, 64)
        self.ll4 = nn.Linear(64, 32)
        self.ll5 = nn.Linear(32, 16)
        self.ll6 = nn.Linear(16, 8)
        self.ll7 = nn.Linear(8, 4)
        self.ll8 = nn.Linear(4, 2)
        self.ll9 = nn.Linear(2, 1)
        self.leaky_relu = nn.LeakyReLU(0.15)

    def forward(self, x):
        x = self.leaky_relu(self.ll(x))
        x = self.leaky_relu(self.ll2(x))
        x = self.leaky_relu(self.ll21(x))
        x = self.leaky_relu(self.ll22(x))
        x = self.leaky_relu(self.ll3(x))
        x = self.leaky_relu(self.ll4(x))
        x = self.leaky_relu(self.ll5(x))
        x = self.leaky_relu(self.ll6(x))
        x = self.leaky_relu(self.ll7(x))
        x = self.leaky_relu(self.ll8(x))
        x = self.leaky_relu(self.ll9(x))
        x = torch.sigmoid(x)
        return x


class MyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features.iloc[index, :].values, self.targets.iloc[index]
