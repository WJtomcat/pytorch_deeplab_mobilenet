import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import to_np, label_accuracy_score
from dataset.voc import VOC2012ClassSeg
from model import Deeplab

from logger import Logger

model = Deeplab()

base_params = list(map(id, model.base_net.parameters()))
top_params = filter(lambda p: id(p) not in base_params,
                    model.parameters())

# for param in top_params:
#     print(type(param.data))

for param in base_params:
    print(type(param))

# optimizer = torch.optim.SGD([
#     {'params': top_params},
#     {'params': base_params, 'lr': lr * 0.01}],
#     lr=lr, momentum=0.9, weight_decay=0.00004)
