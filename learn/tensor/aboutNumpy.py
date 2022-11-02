import numpy as np
import torch

'''
张量和numpy
'''
# 在cpu上对的张量和numpy数据共享他们的内存位置，改变一个就会改变另一个
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 改张量的值，numpy数据的值也随之改变
t.add_(1)

print(f"t: {t}")
print(f"n: {n}")

'''
numpy数组转换成张量
'''
np1 = np.ones(5)
print(f"n: {np1}")
tp1 = torch.from_numpy(np1)
print(f"t: {tp1}")

np.add(np1, 2, out=np1)
print(f"n: {np1}")
print(f"t: {tp1}")









