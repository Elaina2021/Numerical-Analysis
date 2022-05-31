# Time: 2022/5/24  11:41
import torch
import torch.nn as nn
import torch.nn.functional as F

class SNet(nn.Module): # 要继承nn.Module的父类
    def __init__(self):
        super(SNet, self).__init__() # 涉及多继承，调用父类的构造函数
        self.fc1 = nn.Linear(2, 3) # 展平后做线性变换
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1) # 最后的输出个数是10个，10个类别

    def forward(self, x): # 正向传播过程
        # x = x.view(-1, 32*5*5)       # output(32*5*5) # 展平成一维
        x = torch.sigmoid(self.fc1(x))      # output(120)
        x = torch.sigmoid(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10) # 最后的输出只是线性变换，没有softmax
        return x


# test

# model = SNet()
# x = torch.rand(2)
# y = model(x)
# print(x)
# print(y)
