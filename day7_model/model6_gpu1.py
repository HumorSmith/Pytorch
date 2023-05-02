import torch.optim
import torchvision.datasets
import time
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from day7_model.model import CIFAR10NN

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset_test", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集的长度为:{}".format(train_data_size))
print("测试集的长度为:{}".format(test_data_size))

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

cifar10nn = CIFAR10NN()
if torch.cuda.is_available():
    cifar10nn.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn.cuda()

# 优化器
# 1e-2 = 1*(10)^(-2) = 1/100 = 0.01
learning_rate = 0.01
optimizer = torch.optim.SGD(cifar10nn.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 总体准确率

# 训练的轮数
epoch = 10

# 添加tensorboard
summy = SummaryWriter("./train_logs")
cifar10nn.train()
start_time = time.time()
for i in range(epoch):
    print("----------第{}轮训练开始------------".format(i))
    for data in train_data_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs.cuda()
            targets.cuda()
        output = cifar10nn(imgs)
        # 优化模型
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, loss:{}".format(total_train_step, loss.item()))
            summy.add_scalar("train_loss", loss.item(), total_train_step)

    cifar10nn.eval()
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs.cuda()
                targets.cuda()
            outputs = cifar10nn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            total_test_step = total_test_step + 1
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
        # print("测试次数:{}".format(total_test_step))
    summy.add_scalar("test_loss", total_test_loss, total_test_step)
    summy.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体的正确率{}".format(total_accuracy / test_data_size))
    torch.save(cifar10nn, "cifar10nn_{}.pth".format(i))
