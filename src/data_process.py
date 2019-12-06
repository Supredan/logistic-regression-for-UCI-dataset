import random

import torch
import torch.nn as nn
import numpy as np
torch.__version__

from torch.utils.data import Dataset
import pandas as pd

data = pd.read_csv('../data/adult.data', header=None)

row = data[0:1]
d = {}
for index in row:
    temp = row[index]
    if temp.dtype != int:
        keys = list(set(data[index]))
        values = range(len(keys))
        d.update(dict(zip(keys, values)))
        # print(dict(zip(keys, values)))
        # for index_col in data[index].keys():
        #     data.loc[index_col, index] = d[data[index][index_col]]

data = data.applymap(lambda x: d[x] if type(x) != int else x)
data.to_csv('../data/PreProcess_adult.data', header=None, index=None)
d.update({' <=50K.': d[' <=50K'], ' >50K.': d[' >50K']})

data = pd.read_csv('../data/adult.test', header=None)
data = data.applymap(lambda x: d[x] if type(x) != int else x)
data.to_csv('../data/PreProcess_adult.test', header=None, index=None)