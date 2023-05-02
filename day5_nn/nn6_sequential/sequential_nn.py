from torch import nn
from torch.nn import ReLU, Sequential, Conv2d, MaxPool2d, Flatten, Linear


class SequentialNN(nn.Module):
    def __init__(self):
        super(SequentialNN, self).__init__()
        self.mode1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(5, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.mode1(input)
        return output
