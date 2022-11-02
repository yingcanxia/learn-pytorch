'''
随机生成3张大小为28*28的图像，小批量样本，观察每一层对输入数据处理的结果
'''

import torch
from torch import nn



input_image = torch.rand(3,28,28)
print(input_image.size())

'''
nn.Flatten
压平
将每个图片为28*28的图像转换为784个像素值的连续数组
保持批量维度为0
'''


flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())


'''
nn.Linear
线性层事故听其存储的权重w和偏差b对输入应用进行线性转换
'''
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

'''
nn.ReLU
在线性变换后应用以引入非线性，磅数神经网络学习各种想先
在线性层中使用nn.ReLU但还需要其他非线性激活函数
'''

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

'''
nn.Sequential
可以理解为网络层的容器，在骑宠我们定义各种网络，数据会按照我们设置的顺序经过所有网络层
'''
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

logits = seq_modules(input_image)

'''
nn.Softmax
神经网络的最后一个线性层返回的logits，取值为[-infty,infty]。经过nn.Softmax函数之后，logits的值收敛到[0,1],标识模型对每个类别的预测概率。
dim参数指示值必须总和为1的维度
'''
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

