
from model import Deeplab
import torch
import torch.nn as nn

y = torch.load('./convert_basenet/mobilenetv2_9.pth')

def weights_normal_init(model):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

model = Deeplab()
weights_normal_init(model)
x = model.state_dict()

i=0
for k in list(x.keys()):
    if 'base_net' in k:
        x[k] = y['module.' + k[k.find('.')+1:]]
        i += 1

torch.save(x, 'deeplab_init_9.pth')
