import numpy as np
import torch
from torch import nn
from torch.utils import data

"""数据测试"""
test_data = torch.load('test-file')
test_iter = data.DataLoader(test_data)

net = nn.Sequential(nn.Linear(9, 7).double(), nn.Sigmoid(),
                    nn.Linear(7, 5).double(), nn.Sigmoid(),
                    nn.Linear(5, 4).double(), nn.Sigmoid(),
                    nn.Linear(4, 3).double(), nn.Sigmoid(),
                    nn.Linear(3, 2).double())
net.load_state_dict(torch.load('net_params-file'))
net.eval() # 测试模式
with torch.no_grad():
    for i, (X, y) in enumerate(test_iter):
        y_hat = net(X)            
        print(y_hat.numpy(), y.numpy())
