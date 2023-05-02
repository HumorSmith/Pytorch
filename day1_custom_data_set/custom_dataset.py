import os

from torch.utils.data.dataset import Dataset


class CustomDataSet(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        # 获取文件列表
        self.list_dir = os.listdir(self.img_dir)

    def __len__(self):
        len(self.list_dir)

    def __getitem__(self, index):
        self.item_name = self.list_dir[index]
        self.item_image_path = os.path.join(self.img_dir, self.item_name)
        self.item_label_path = os.path.join(self.label_dir, self.item_name)
        return self.item_image_path, self.item_label_path
