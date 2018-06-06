
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import to_np, label_accuracy_score, fast_hist
from dataset.voc import VOC2012ClassSeg
from model import Deeplab

from logger import Logger


def train(epoch_idx, net, train_loader, lr, logger, n_class):
    net.cuda()
    net.train()

    base_params = list(map(id, net.base_net.parameters()))
    top_params = filter(lambda p: id(p) not in base_params,
                        net.parameters())

    optimizer = torch.optim.SGD([
        {'params': top_params},
        {'params': net.base_net.parameters(), 'lr': lr * 0.1}],
        lr=lr, momentum=0.9, weight_decay=0.00004)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    len_batch = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        score = net(data)
        loss = criterion(score, target)
        loss.backward()
        optimizer.step()

        _, predicted = score.max(1)
        predicted, target = to_np(predicted), to_np(target)
        acc, acc_cls, mean_iu = label_accuracy_score(target, predicted, n_class)
        info = {
            'acc': acc,
            'acc_cls': acc_cls,
            'mean_iu': mean_iu,
            'loss': loss.data[0]
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, len_batch*epoch_idx+batch_idx+1)
        print(('train', batch_idx, epoch_idx))

    if (epoch_idx+1) % 10 == 0:
        n = (epoch_idx+1) / 10
        state = net.state_dict()
        torch.save(state, './deeplab_epoch_' + str(n)+ '.pth')

def test(epoch_idx, net, test_loader, logger, n_class):
    net.cuda()
    net.eval()
    len_batch = len(test_loader)

    visualizations = []

    hist = np.zeros((n_class, n_class))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            output = net(inputs)
            _, predicted = output.max(1)
            predicted, targets = to_np(predicted), to_np(targets)
            print(('test', batch_idx, epoch_idx))
            hist += fast_hist(targets.flatten(), predicted.flatten(), n_class)
        miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        miou = np.sum(miou)/len(miou)
        logger.scalar_summary('Mean iou', miou, epoch_idx)
        print(('Mean iou: ', miou))




def main():

    logger = Logger('./logs')

    net = Deeplab()

    train_dataset = VOC2012ClassSeg('./dataset', split='train', transform=True,
                                    is_training=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    test_dataset = VOC2012ClassSeg('./dataset', split='val', transform=True,
                                   is_training=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    n_class = len(train_dataset.class_names)

    model_file = './deeplab.pth'
    model_data = torch.load(model_file)
    net.load_state_dict(model_data)

    lr = 0.001

    for epoch_idx in range(100):
        test(epoch_idx, net, test_loader, logger, n_class)
        train(epoch_idx, net, train_loader, lr, logger, n_class)
        lr *= 0.9

main()
