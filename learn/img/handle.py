import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

local_download_url = "D://pytorchDataset"
# 原始的数据格式不一定符合模型训练多需要的输入格式，于是就需要通过torchvision.transforms来对数据进行依稀操作并使其适合训练
ds = datasets.FashionMNIST(
    root=local_download_url,
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)
'''
transfroms.ToTensor()
将PLI image或者numpy.ndarray格式的数据转换成tensor格式，使响度大小多放到0,1

transforms.Normalize()
对输入进行标准化，传入均值和标准差与输入的维度相同

transforms.ToPILImage()
将tensor或者numpy.ndarray格式的数据转换为PIL Image图片格式。

transforms.Resize()
修改图片的尺寸，参数size可以是序列也可以是整数，如果传入序列，则修改后的图片尺寸和序列一直，如果传入整数则等比例缩放图片

transforms.CenterCrop()
中心裁剪图片，参数size，可以是序列也可以是整数，如果传入序列，则裁剪后的图片尺寸和序列一样，如果是整数，则裁剪尺寸查款都是size的正方形

transforms.RandomCrop()
堆积建材，参数size可以是序列也可以是整数，如果插传入序列，则裁剪后的图片尺寸和序列一致，如果传入整数，则长款都是size的正方形

transforms.RandomResizedCrop()
将给定图像随件裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定的大小

transforms.RandomHorizontalFlip()
有一定概率将图片水平翻转，默认概率为0.5

transforms.RandomVerticalFlip()
有一定概率将图片垂直翻转，默认概率为0.5

transforms.RandomRotation()
将图片翻转，参数degrees可以为序列或者数值，如果为序列，则先转角度为(min_degree, max_degree);如果为数值则旋转角度为-degrees，+degrees
'''