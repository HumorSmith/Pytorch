import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ChannelModule(nn.Module):
    def __init__(self):
        super(ChannelModule, self).__init__()
        self.conv = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        output = self.conv(input)
        return output


dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)
channelModule = ChannelModule()
summy = SummaryWriter("channel_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    summy.add_images("input", imgs, step)
    output = channelModule(imgs)
    output = torch.reshape(output, (-1, 3, 30, 30))
    summy.add_images("output", output, step)
    step = step + 1
summy.close()
