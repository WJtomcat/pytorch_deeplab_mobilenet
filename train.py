
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

def main():

  logger = Logger('./logs')

  model = Deeplab()

  dataset = VOC2012ClassSeg('./dataset', split='train', transform=True)

  train_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=2,
      shuffle=True,
      num_workers=1)

  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  criterion = nn.CrossEntropyLoss(ignore_index=-1)

  n_class = len(dataset.class_names)

  model_file = 'deeplab_init.pth'
  moda_data = torch.load(model_file)


  model.cuda()

  model.train()

  label_trues, label_preds = [], []

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()

    score = model(data)
    loss = criterion(score, target)
    loss.backward()
    optimizer.step()

    _, predicted = score.max(1)
    predicted, target = to_np(predicted), to_np(target)
    print(predicted)
    print(label_accuracy_score(predicted, target, n_class))
    break




main()
