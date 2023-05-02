import torchvision.models
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 增加网络
vgg16_true.add_module("add_linear", nn.Linear(1000, 10))

# 在classifier里面加
vgg16_true.classifier.add_module("add_classifier", nn.Linear(1000, 10))
print(vgg16_true)

# 修改classifier第六个元素
vgg16_true.classifier[6] = nn.Linear(4096, 10)
