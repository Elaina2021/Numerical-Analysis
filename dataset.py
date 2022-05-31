import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MyDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self,root=None,train=True):
        self.data = pd.read_csv(root).values

        if train == True:
            self.x = torch.from_numpy(self.data[:80,:2]).float()
            self.y = torch.from_numpy(self.data[:80,2:]).float()
            self.len = self.y.shape[0]

        if train == False:
            self.x = torch.from_numpy(self.data[80:,:2]).float()
            self.y = torch.from_numpy(self.data[80:,2:]).float()
            self.len = self.y.shape[0]
    # 要重写两个函数
    def __getitem__(self, index):
        # return torch.from_numpy(self.x[index],dtype=torch.float), torch.from_numpy(self.y[index],dtype=torch.float)

        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

# test

#
# mydataset = MyDataset(root='./Student_Marks.csv',train=True)
# print('x: ', mydataset.x.dtype, 'y: ', mydataset.y)
# print('len: ', mydataset.len)



