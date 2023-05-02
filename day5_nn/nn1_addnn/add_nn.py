import torch
from torch import nn


class AddNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


add_nn = AddNN()
tensor = torch.tensor(1.0)
output = add_nn(tensor)
print(output)
