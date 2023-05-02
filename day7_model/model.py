import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


class CIFAR10NN(nn.Module):
    def  __init__(self):
        super(CIFAR10NN, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output


if __name__ == '__main__':
    model = CIFAR10NN()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)
