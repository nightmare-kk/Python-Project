import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


# 随机种子
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# 导入数据
train_data, test_data = datasets.load_dataset("stanfordnlp/imdb", split=['train', 'test'])

print(train_data, test_data)