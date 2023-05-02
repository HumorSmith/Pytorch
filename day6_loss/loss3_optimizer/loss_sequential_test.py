import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

from day5_nn.nn6_sequential.sequential_nn import SequentialNN

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

sequential_nn = SequentialNN()

optim = torch.optim.SGD(sequential_nn.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
running_loss = 0.0
for epoch in range(20):
    for data in dataloader:
        imgs, targets = data
        output = sequential_nn(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        print(output)
        print(imgs)
        running_loss = running_loss + result_loss
    print(running_loss)
