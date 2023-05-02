import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from day5_nn.nn5_no_linear_activate.reLU.relu_nn import ReLUNN

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

step = 0
summy = SummaryWriter("relu_logs")
relu = ReLUNN()
for data in dataloader:
    imgs, targets = data
    summy.add_images("input", imgs, global_step=step)
    output = relu(imgs)
    summy.add_images("output", output, global_step=step)
    step = step + 1

# num test
numTensor = torch.tensor([[-1, -1], [1, 2]])

numTensor = torch.reshape(numTensor, (1, 1, 2, 2))
output = relu(numTensor)
print(output)

summy.close()
