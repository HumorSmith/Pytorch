import torchvision.datasets

from day1_custom_data_set.custom_dataset import CustomDataSet

# 自定义data set
customDataSet = CustomDataSet(img_dir="./hymenoptera_data/train/ants", label_dir="ants")
print(customDataSet[0])

# 下载dataset
dataset = torchvision.datasets.CIFAR10(root="./dataset", download=True)
print(dataset[0])
