import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path                                                          #path="D:pythonSpace\data\VOC\VOCdevkit\VOC2007"是初始化地址
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))           #path拼接'SegmentationClass'，再获取下面所有文件名

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):                                                 #制作数据
        segment_name = self.name[index]                                           #名字格式是xx.png，但原图是jpg，故要转化，制作标签
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name) #得到标签地址
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png','jpg'))    #得到原图地址#网络输入需要固定大小的图片，故要等比缩放（贴到mask上），把这个操作写成工具包utils
        segment_image = keep_image_size_open(segment_path)                        
        image = keep_image_size_open_rgb(image_path)
        return transform(image), torch.Tensor(np.array(segment_image))


if __name__ == '__main__':
    from torch.nn.functional import one_hot
    data = MyDataset('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
