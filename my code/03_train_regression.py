import torch
from torch import nn
from torch.utils import data

class Accumulator:
    """累加器"""
    def __init__(self, n): # n个变量
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def test(net, test_iter, loss):
    metric = Accumulator(2) # 2个变量分别为 测试损失之和 样本数
    net.eval() # 测试模式
    with torch.no_grad():
        for i, (X, y) in enumerate(test_iter):
            y_hat = net(X)            
            l = loss(y_hat, y)
            metric.add(l * X.shape[0], X.shape[0])
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr):
    """训练模型"""
    def init_weights(m):
        """初始化"""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    for epoch in range(num_epochs):
        metric = Accumulator(2) # 2个变量分别为 训练损失之和 样本数
        net.train() # 训练模式
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            y_hat = net(X)            
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], X.shape[0])
        train_l = metric[0] / metric[1] # 训练平均损失
        test_l = test(net, test_iter, loss) # 测试平均损失
        print(f'epoch {epoch+1}, train loss {train_l:.3f}, test loss {test_l:.3f}')

"""数据训练"""
batch_size = 10000
train_data = torch.load('train-file')
test_data = torch.load('test-file')
train_iter = data.DataLoader(train_data, batch_size)
test_iter = data.DataLoader(test_data, len(test_data))

net = nn.Sequential(nn.Linear(9, 32).double(), nn.ReLU(),
                    nn.Linear(32, 16).double(), nn.ReLU(),
                    nn.Linear(16, 8).double(), nn.ReLU(),
                    nn.Linear(8, 4).double(), nn.ReLU(),
                    nn.Linear(4, 4).double()) # 试图增加复杂度
lr, num_epochs = 0.01, 20
train(net, train_iter, test_iter, num_epochs, lr)
torch.save(net.state_dict(), 'net_params-file')