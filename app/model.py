import torch.nn as nn

class mnist_model(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
        )
        self.block2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
        )
        self.block3 = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x