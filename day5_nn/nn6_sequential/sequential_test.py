import torch
import torchvision.datasets
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# num test
numTensor = torch.ones((64, 3, 32))

sequential = Sequential()
output = sequential(sequential)
print(sequential)
summy = SummaryWriter("./logs")
summy.add_graph(sequential, numTensor)
summy.close()
