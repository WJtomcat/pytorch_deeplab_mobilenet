
import os
import os.path as osp

import numpy as np
import torch
from torch.autograd import Variable

from utils import to_np, label_accuracy_score
from dataset.voc import VOC2012ClassSeg
from model import Deeplab

def main():

  model = Deeplab()

  dataset = VOC2012ClassSeg('./dataset', split='val', transform=True)

  val_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True)

  # n_class = len(dataset.class_names)

  # model_file = ''
  # moda_data = torch.load(model_file)
  # try:
  #   model.load_state_dict(model_data)
  # except Exception:
  #   model.load_state_dict(model_data['model_state_dict'])
  # if torch.cuda.is_available():
  #   model.cuda()

  model.eval()

  label_trues, label_preds = [], []

  for batch_idx, (data, target) in enumerate(val_loader):

    # if torch.cuda.is_available():
    #   data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    score = model(data)
    _, predicted = score.max(1)
    predicted = to_np(predicted)
    target = to_np(target)
    for lt, lp in zip(target, predicted):
      label_trues.append(lt)
      label_preds.append(lp)
    if batch_idx == 5:
      break
  n_class = 21
  print(len(label_preds))
  metrics = label_accuracy_score(label_trues, label_preds, n_class=n_class)
  metrics = np.array(metrics)
  metrics *= 100
  print(metrics)










    # imgs = to_np(imgs)

main()
