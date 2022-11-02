import torch

'''
索引与切片
'''
# 获取一个4*4的tensor
tensor = torch.ones(4, 4)
# 打印第一行
print(f"First row: {tensor[0]}")
# 打印第一列
print(f"First column: {tensor[:, 0]}")
# 打印最后一列
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

'''
张量的链接
'''
# 可以通过torch.cat()或者torch.stack()来拼接张量
# 在第一个维度进行拼接，也就是水平方向
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

'''
算数运算
'''
print("矩阵乘法")
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)


# 矩阵逐元素相乘，z1、z2和z3的值相同
print("逐个乘法")
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)

'''
单元素张量
只有一个值的张量，可以通过item属性转换为数值
'''
agg = tensor.sum()
agg_item = agg.item()
print(agg)
print(agg_item)
print(type(agg_item))

'''
就地操作，相当于是变量变更后直接调整变量，原值将不再保留
'''
tensor.add_(5)
