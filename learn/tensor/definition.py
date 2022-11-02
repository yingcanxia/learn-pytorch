import torch
import numpy as np

# 从数据中创建，自动推断数据类型
'''
torch.tensor() 直接创建命令
参数含义：
data： 数据
dtype: 数据类型
device:所在设备
require_grad: 是否需要梯度
pin_memory：是否需要锁页内存
'''
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data, dtype=torch.int, device="cuda")
print(f"Tensor from Data:\n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from Numpy:\n {x_np} \n")

'''
根据另一个张量创建
'''
# 保留原有张量的形状和数据类型
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

# 显式更改张量的数据类型
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

'''
使用随机或恒定值创建
shape是张量维度的元组，它决定了输出张量的形状。
'''
# 创建2行3列的张量 指定形状
shape = (2, 3,)

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



