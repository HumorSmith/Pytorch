import torch
from torch.nn import L1Loss, MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)
print(result)

# 更变成sum对比
loss = L1Loss(reduction="sum")
result = loss(inputs, targets)
print(result)

# 平方差
loss = MSELoss()
result = loss(inputs, targets)

print(result)
