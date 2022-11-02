import os
import torch
from torch import nn

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

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

