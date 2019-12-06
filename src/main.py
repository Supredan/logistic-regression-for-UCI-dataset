import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.__version__

from torch.utils.data import Dataset
import pandas as pd

# 数据的生成
train = pd.read_csv('../data/PreProcess_adult.data', header=None)
test = pd.read_csv('../data/PreProcess_adult.test', header=None)

train = np.array(train)
# train = torch.tensor(train)

test = np.array(test)
# test = torch.tensor(test)

n,l=train.shape
for j in range(l-1):
    meanVal=np.mean(train[:,j])
    stdVal=np.std(train[:,j])
    train[:,j]=(train[:,j]-meanVal)/stdVal
np.random.shuffle(train)

n,l=test.shape
for j in range(l-1):
    meanVal=np.mean(test[:,j])
    stdVal=np.std(test[:,j])
    test[:,j]=(test[:,j]-meanVal)/stdVal
np.random.shuffle(test)

train_data = train[:, :14]
train_lab = train[:, 14]

test_data = test[:, :14]
test_lab = test[:, 14]


# 定义模型
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.fc=nn.Linear(14,2) # 由于24个维度已经固定了，所以这里写24
    def forward(self,x):
        x=torch.sigmoid(self.fc(x))
        return x

def test(pred,lab):
    t=pred.max(-1)[1]==lab
    return torch.mean(t.float())

#训练
net=LR()
criterion=nn.CrossEntropyLoss() # 使用CrossEntropyLoss损失
optm=torch.optim.Adam(net.parameters()) # Adam优化
epochs=1000 # 训练1000次

for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    y_hat=net(x)
    loss=criterion(y_hat,y) # 计算损失
    optm.zero_grad() # 前一步的损失清零
    loss.backward() # 反向传播
    optm.step() # 优化
    if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        # 使用我们的测试函数计算准确率
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i+1,loss.item(),accu))
