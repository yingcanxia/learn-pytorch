import torch

tensor = torch.rand(3, 4)


# 将张量移动到GPU上
if torch.cuda.is_available():
    tensor = tensor.to()

# 打印tensor的形状
print(f"Shape of tensor: {tensor.shape}")
# 打印tensor数据类型
print(f"Datatype of tensor: {tensor.dtype}")
# 打印tensor在哪个设备上
print(f"Device tensor is stored on: {tensor.device}")

