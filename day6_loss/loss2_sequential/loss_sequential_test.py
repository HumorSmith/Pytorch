import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

from day5_nn.nn6_sequential.sequential_nn import SequentialNN

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

sequential_nn = SequentialNN()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    output = sequential_nn(imgs)
    result_loss = loss(output, targets)
    result_loss.backward()
    print(output)
    print(imgs)
