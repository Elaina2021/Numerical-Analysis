# Time: 2022/5/24  11:41
import os
import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import SNet
import torch.nn as nn
import torch.optim as optim
import numpy as np

path = './Student_Marks.csv'
train_dataset=MyDataset(path,train=True)
test_dataset=MyDataset(path,train=False)


train_loader = DataLoader(dataset=train_dataset,
                           batch_size=40,
                           shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                           batch_size=20,
                           shuffle=False)

Epoch = 600

net = SNet()
loss_function = nn.MSELoss()  # 这里面会自动包含softmax
optimizer = optim.SGD(net.parameters(), lr=0.01)  # 第一个参数就是要训练的参数，net是刚才定义的，.parameters()就是里面的参数
print("Training...")
for epoch in range(Epoch):  # 训练所有!整套!数据 3 次
    running_loss = 0.0  # 累加损失
    for step,(batch_x,batch_y) in enumerate(train_loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...
        optimizer.zero_grad()
        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 打出来一些数据
        # print('Epoch: ', epoch+1, '| Step:', step+1)

print("Training loss: %.3f" % (loss/80))



print("Testing...")
with torch.no_grad():
    loss = 0
    for step, (batch_x, batch_y) in enumerate(test_loader):

        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss = loss.item()
        table = torch.zeros((2,20))
        table[0] = batch_y.T
        table[1] = outputs.T
        print("Testing loss: %.3f" % (loss/20))
        # print("compare",table)  # 会输出10个值，取其中最大的就是预测的标签


# compare
def MSE_loss(Y_predict,Y):
    n = Y.shape[0]
    # print(n)
    return (((Y_predict-Y)**2).sum())/n

print("MSE loss on test data:",MSE_loss(outputs,batch_y))
print("MAE loss on test data:",MSE_loss(outputs,batch_y).sqrt())
# x = torch.rand((80,2)).float()
# input_names = ["inputs"]
# output_names = ["outputs"]
# # Export the model
# torch_out = torch.onnx.export(net, x, "mymodel.onnx", export_params=True, verbose=True,input_names=input_names, output_names=output_names)





