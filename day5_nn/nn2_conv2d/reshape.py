import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 3], [0, 1, 0], [2, 1, 0]])

input = torch.reshape(input, [1, 1, 5, 5])
kernel = torch.reshape(kernel, [1, 1, 3, 3])
print(input.shape)
print(kernel.shape)

# stride :步进
# padding : 边距,默认padding=0,padding边框的值为0
output = F.conv2d(input, kernel, stride=1, padding=1)
print(output)

# https://www.bilibili.com/video/BV1hE411t7RN
