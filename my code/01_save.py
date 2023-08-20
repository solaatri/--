import numpy as np
import math
import torch

"""数据存储"""
data = np.load('../results.npz', mmap_mode='r')
all_arrays = data['arr_0']
data.close()

mult = all_arrays[:, 0]
ave2_2 = all_arrays[:, 1]
ave4_2 = all_arrays[:, 2]
c24 = ave4_2 - ave2_2 ** 2 # 不独立
ave2_3 = all_arrays[:, 3]
ave4_3 = all_arrays[:, 4]
ave2_4 = all_arrays[:, 5]
ave4_4 = all_arrays[:, 6]
meanpT = all_arrays[:, 7]
flcmeanpT = all_arrays[:, 8]
A_spin = all_arrays[:, 9]*180/math.pi
B_spin = all_arrays[:, 10]*180/math.pi
A_tilt = all_arrays[:, 11]*180/math.pi
B_tilt = all_arrays[:, 12]*180/math.pi


features = all_arrays[:, :9]
labels = all_arrays[:,9:]
features = (features - features.mean(axis = 0)) / features.std(axis = 0) # 标准化
labels = (labels - labels.mean(axis = 0)) / labels.std(axis = 0) # 标准化
torch.save(torch.tensor(features), 'features-file')
torch.save(torch.tensor(labels), 'labels-file')