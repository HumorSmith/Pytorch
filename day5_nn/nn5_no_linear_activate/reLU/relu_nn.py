from torch import nn
from torch.nn import ReLU


class ReLUNN(nn.Module):
    def __init__(self):
        super(ReLUNN, self).__init__()
        self.relu = ReLU()

    def forward(self, input):
        output = self.relu(input)
        return output
