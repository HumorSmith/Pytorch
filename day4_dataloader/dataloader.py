import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=4)
step = 0
summy = SummaryWriter("dataloader")
for item in dataloader:
    imgs, targets = item
    summy.add_images(tag="loader", img_tensor=imgs, global_step=step)
    step = step + 1
summy.close()
