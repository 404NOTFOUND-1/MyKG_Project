import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


# 生成数据
data_tensor = torch.randn(3, 3)
target_tensor = torch.randint(3,(3,2))  # 标签是0或1

# 将数据封装成Dataset
my_dataset = MyDataset(data_tensor, target_tensor)
print(data_tensor)
print(target_tensor)
# 查看数据集大小
print('dataset[0]', my_dataset[0])
print('dataset[1]', my_dataset[1])
