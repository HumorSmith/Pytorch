import torch
import torchvision.transforms
from PIL import Image

from day7_model.model import CIFAR10NN

image_path = "./dog/dogs.png"
image = Image.open(image_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = image.convert("RGB")
image = transform(image)
print(image.shape)

cifar10nn = CIFAR10NN()
print(cifar10nn)
# 直接加载，不用赋值
cifar10nn.load_state_dict(torch.load("./cifar10nn_30.pth", map_location=torch.device('cpu')))
print(cifar10nn)
image = torch.reshape(image, (1, 3, 32, 32))
with torch.no_grad():
    output = cifar10nn(image)
print(output)
print(output.argmax(1))
