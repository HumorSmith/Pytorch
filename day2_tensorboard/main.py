import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

summy = SummaryWriter("logs")
for i in range(100):
    summy.add_scalar("y = x", i, i)
img = Image.open("./img/0013035.jpg")
img_arr = np.asarray(img)
print(img_arr.shape)
print(img_arr.dtype)

# dataformats这里可以通过打印np获得，HW代表高宽，C代表channel
summy.add_image("ant", img_arr, 1, dataformats="HWC")
summy.close()
# 查看
