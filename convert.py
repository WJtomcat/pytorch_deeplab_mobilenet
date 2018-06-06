from model import Deeplab
import tensorflow as tf
from collections import OrderedDict
import torch

model = Deeplab()
x = model.state_dict()

model_name = './convert_model/deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000'

reader = tf.train.NewCheckpointReader(model_name)
var_to_shape_map = reader.get_variable_to_shape_map()
var_dict = {k:reader.get_tensor(k) for k in var_to_shape_map.keys()}

for k in list(var_dict.keys()):
    if 'RMSProp' in k or 'Momentum' in k or 'ExponentialMovingAverage' in k or 'global_step' in k:
        del var_dict[k]

# i = 0
# for k in list(x.keys()):
#     if 'base_net' in k:
#         i += 1
# print(i)
#
# i = 0
# for k in list(var_dict.keys()):
#     if 'MobilenetV2' in k:
#         i += 1
# print(i)

for k in list(var_dict.keys()):
  if 'MobilenetV2' in k:
    var_dict['base_net.layers.'+k[k.find('/')+1:]] = var_dict[k]
    del var_dict[k]

dummy_replace = OrderedDict([
    ('expanded_conv_', ''),
    ('expanded_conv', '0'),
    ('/depthwise/depthwise_weights', '.conv2.weight'),
    ('/depthwise/BatchNorm', '.bn2'),
    ('/project/weights', '.conv3.weight'),
    ('/project/BatchNorm', '.bn3'),
    ('/expand/BatchNorm', '.bn1'),
    ('/expand/weights', '.conv1.weight'),
    ('/moving_mean', '.running_mean'),
    ('/moving_variance', '.running_var'),
    ('/gamma', '.weight'),
    ('/beta', '.bias'),
    ('image_pooling/weights', 'avgpool_connect.weight'),
    ('image_pooling/BatchNorm', 'avgpool_batchnorm'),
    ('layers.Conv/weights', 'conv1.weight'),
    ('layers.Conv/BatchNorm', 'bn1'),
    ('aspp0/weights', 'head_conv.weight'),
    ('aspp0/BatchNorm', 'head_batchnorm'),
    ('concat_projection/weights', 'top_conv.weight'),
    ('concat_projection/BatchNorm', 'top_batchnorm'),
    ('logits/semantic/weights', 'semantic_conv.weight'),
    ('logits/semantic/biases', 'semantic_conv.bias'),
])
for a, b in dummy_replace.items():
    for k in list(var_dict.keys()):
        if a in k:
            var_dict[k.replace(a,b)] = var_dict[k]
            del var_dict[k]

# for k in list(var_dict.keys()):
#     print(k)
#
# print('----------------------------')
#
# y = set(var_dict.keys()) - set(x.keys())
# for k in y:
#     print(k)
#
# print('----------------------------')
#
# y = set(x.keys()) - set(var_dict.keys())
# for k in y:
#     print(k)
#
# print(len(y))
for k in list(var_dict.keys()):
    if var_dict[k].ndim == 4:
        if '.conv2' in k:
            var_dict[k] = var_dict[k].transpose((2, 3, 0, 1)).copy(order='C')
        else:
            var_dict[k] = var_dict[k].transpose((3, 2, 0, 1)).copy(order='C')
    if var_dict[k].ndim == 2:
        var_dict[k] = var_dict[k].transpose((1, 0)).copy(order='C')
    if x[k].shape != var_dict[k].shape:
        print(k)
        print(x[k].shape)
        print(var_dict[k].shape)

for k in list(var_dict.keys()):
    var_dict[k] = torch.from_numpy(var_dict[k])

torch.save(var_dict, 'deeplab.pth')
