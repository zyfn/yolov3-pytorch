import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from PIL import ImageFile
import os
# try:

#     img_path = '/d/Code/PyTorch-YOLOv3/data/coco/images/train2014/COCO_train2014_000000006586.jpg'
#     temp=list(img_path)
#     del temp[0]
#     temp.insert(1,':')
#     img_path = ''.join(temp)
#     print(img_path)
#     # img_path = 'd:\\Code\\PyTorch-YOLOv3\\data\\coco\\images\\train2014\\COCO_train2014_000000006586.jpg'
#     # print(os.path.exists(img_path))


#     img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
# except Exception:
#     print(f"Could not read image '{img_path}'.")

path='D:\Code\PyTorch-YOLOv3\data\custom\images\\val\\'
datanames = os.listdir(path)
for i in datanames:
    with open('D:\Code\PyTorch-YOLOv3\data\custom\\val.txt', "a") as f:
        f.write(f'{path}{i}\n')
    f.close()
