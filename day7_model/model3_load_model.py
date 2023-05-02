import torch

import torchvision.models

# 方式1,需要带上模型类
model = torch.load("vgg16_method1.pth")

# 方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
# 直接加载，不用赋值回去
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
