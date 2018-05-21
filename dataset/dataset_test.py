
from voc import VOC2012ClassSeg

import torch

root = './'
split = 'train'


dataset = VOC2012ClassSeg(root, split=split)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=10, shuffle=True, transform=True)

print(len(train_loader))

for batch_idx, (data, target) in enumerate(train_loader):
  print(data.shape)
  print(target.shape)
  break
