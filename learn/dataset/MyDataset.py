import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class MyImageDateset(Dataset):

    def __init__(self, annotations_file,img_dir, transform=None, target_transform=None):
        # 读取文件标签
        self.img_labels = pd.read_csv(annotations_file)
        # 读取图片存储路径
        self.img_dir = img_dir
        # 数据处理方法
        self.transform = transform
        # 标签处理方式
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 单张图片的路径
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # 读取图片
        image = read_image(img_path)
        # 获取对应的标签
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label