'''
torch.nn提供了构建神经网络所需的全部模块
接下来构建一个神经网络对FashionMNIST数据集中的图像进行分裂
'''

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 判定环境
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

'''
模型的定义需要继承基类torch.nn.Module,__init__函数初始化网络模型中的各种层，forward函数对输入数据进行响应的操作
'''

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 实例化NeuralNetwork类，并将其移动到device上。
model = NeuralNetwork().to(device)
print(model)
'''
我们可以将输入数据传入模型，会自动调用forward方法，模型会返回一个10维的张量，其中包含每个累的原始预测值，使用nn.Softmax函数来预测类别的概率
'''
X = torch.rand(1, 28, 28, device=device)
# 调用forward函数
logits = model(X)
# 在第一个维度应用Softmax函数
pred_probab = nn.Softmax(dim=1)(logits)
# 最大概率值对应的下标
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
