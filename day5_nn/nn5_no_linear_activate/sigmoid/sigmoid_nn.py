from torch import nn
from torch.nn import ReLU, Sigmoid


class SigmoidNN(nn.Module):
    def __init__(self):
        super(SigmoidNN, self).__init__()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output
