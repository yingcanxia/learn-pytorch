import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# 从numpy创建张量
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from Numpy:\n {x_np} \n")

# 根据一个张量创建另一个张量
# 保留原来的形状然后全都是1
x_one = torch.ones_like(x_data)
# 显式更改张量的数据类型
x_rand = torch.rand_like(x_data, dtype=torch.float)

# 使用随机或者恒定值创建
# shape是张量维度的元祖， 它决定了输出张量的形状
# 创建两行2行3列的张量
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 张量的属性
# 张量的形状，数据类型还有存储设备
tensor = torch.rand(3, 5)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 张量的操作
# 张量有多种高功能，包括算数，线性代数，矩阵操作，采样等等且可以使用gpu
# 默认情况下张量在CPU上创建，如果需要的话，可以使用.to方法明确的量张量移动到GPU上，当然是GPU可用的情况下
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")

# 索引与切片
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# 链接张量
# 在第1个维度拼接，也就是水平方向
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 算数运算
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# 逐个元素想成
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)



