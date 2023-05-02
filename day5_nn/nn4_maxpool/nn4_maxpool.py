import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from day5_nn.nn4_maxpool.max2dpool_module import Max2dPoolNN

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

step = 0
max2dPool = Max2dPoolNN()
summy = SummaryWriter("max_pool")
for data in dataloader:
    imgs, targets = data
    summy.add_images("input", imgs, global_step=step)
    output = max2dPool(imgs)
    summy.add_images("output", output, global_step=step)

summy.close()
