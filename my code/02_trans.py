import torch
from torch.utils import data

def train_test_iter(data_arrays, train_per, batch_size, data_size):
    """构造训练测试数据集"""
    dataset = data.TensorDataset(*data_arrays)
    return torch.utils.data.random_split(dataset, [int(train_per*len(dataset)), len(dataset)-int(train_per*len(dataset))])

"""数据转换"""
features, labels = torch.load('features-file'), torch.load('labels-file')
data_size = features.shape[0]
train_per, batch_size = 0.9, 256
train_data, test_data = train_test_iter((features, labels), train_per, batch_size, data_size)
torch.save(train_data, 'train-file')
torch.save(test_data, 'test-file')