from torch import nn
from torch.nn import MaxPool2d


class Max2dPoolNN(nn.Module):
    def __init__(self):
        super(Max2dPoolNN, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output
