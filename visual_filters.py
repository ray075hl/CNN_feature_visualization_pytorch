import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.vgg import vgg16

import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

from utils import deprocess_image

# https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94/6
grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


vgg = vgg16(pretrained=True)

num_conv = 1
num_block = 1
conv_layer = {}

# Parse each conv layer's position
for i, layer in enumerate(vgg.features.children()):
    if isinstance(layer, nn.Conv2d):
        conv_layer['block{}_conv{}'.format(num_block, num_conv)] = i
        num_conv += 1

    if isinstance(layer, nn.MaxPool2d):
        num_conv = 1
        num_block += 1

print(conv_layer)


class VfModel(nn.Module):
    def __init__(self, pretrained_model, layer_name):
        super(VfModel, self).__init__()
        self.feature_model = torch.nn.Sequential(*list(pretrained_model.features.children())[:conv_layer[layer_name]+1])

    def forward(self, x):
        f = self.feature_model(x)
        return f


net = VfModel(vgg, 'block4_conv1')

size = 150
input_img = Variable(torch.randn(1, 3, size, size) * 20 + 128., requires_grad=True)
filters_num = net(input_img).size()[1]

filters_num = 8
image_per_row = 4
rows = filters_num // image_per_row
display_grad = np.zeros((size * rows, image_per_row * size, 3))

step = 1.0
for filter_index in range(filters_num):
    # input_img = copy.deepcopy(Variable(torch.randn(1, 3, size, size) * 20 + 128., requires_grad=True))
    row = filter_index // image_per_row
    col = filter_index - row * image_per_row
    # Gradient ascent
    for i in range(40):

        net.zero_grad()

        feature = net(input_img)

        feature0 = feature[:, filter_index, :, :]
        loss = torch.mean(feature0)

        input_img.register_hook(save_grad('input_img'))
        loss.backward()

        grads_ = grads['input_img']
        # print(grads.size())
        grads = {}
        input_img = input_img + grads_ * step

    print(filter_index, input_img.size())
    output_img = deprocess_image(input_img.squeeze().cpu().detach().numpy())

    display_grad[row*size : (row+1)*size, col*size: (col+1)*size, :] = output_img

scale = 1. / size
plt.figure(figsize=(scale * display_grad.shape[1], scale * display_grad.shape[0]))
plt.imshow(display_grad/255.0, aspect='auto', cmap='viridis')
plt.show()