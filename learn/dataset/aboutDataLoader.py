import MyDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


'''
dataset 定义好的数据集
batch_size 每次放入网络训练的批次大小
shuffle 是否打乱数据的顺序，默认False 一般训练集设置为true，但是测试集设置为False
droop_last 是否丢弃最后一个批次的数据，默认是False
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

data_loader = DataLoader(
    dataset=MyDataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    drop_last=False
)


train_dataloader = DataLoader(
    dataset=training_data,
    # 设置批量大小
    batch_size=64,
    # 打乱样本的顺序
    shuffle=True)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True)


# 展示图片和标签
train_features, train_labels = next(iter(train_dataloader))
# (B,N,H,W)
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# 获取第一张图片，去除第一个批量维度

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
