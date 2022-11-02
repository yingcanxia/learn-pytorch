import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

'''
root是存储训练/测试数据的路径；
train指定训练或测试数据集；
download=True如果本机没有该数据集，则会下载数据到root路径下；

'''

local_download_url = "D://pytorchDataset"
# 训练数据集
training_data = datasets.FashionMNIST(
    root=local_download_url,
    train=True,
    download=True,
    transform=ToTensor()
)
#验证数据集
test_data = datasets.FashionMNIST(
    root=local_download_url,
    train=False,
    download=True,
    transform=ToTensor()
)

'''
可视化数据集
'''
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 设置画布大小
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # 随机生成一个索引
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # 获取样本及其对应的标签
    img, label = training_data[sample_idx]
    # 添加子图
    figure.add_subplot(rows, cols, i)
    # 设置标题
    plt.title(labels_map[label])
    # 不显示坐标轴
    plt.axis("off")
    # 显示灰度图
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
