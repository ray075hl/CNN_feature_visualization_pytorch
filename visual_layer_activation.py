import sys

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

import cv2
import copy
import numpy as np

import matplotlib.pyplot as plt

from utils import Show_origin_and_output, preprocess_input

vgg = vgg16(pretrained=True)

print(vgg)
CONV_LAYERS = [1, 3, 6, 8, 11, 13, 15]


class VaModel(nn.Module):
    def __init__(self, pretrained_model, conv_index=0):
        super(VaModel, self).__init__()
        self.conv_feature = torch.nn.Sequential(*list(pretrained_model.features.children())[:CONV_LAYERS[conv_index]])

    def forward(self, x):
        batch = x.size()[0]
        conv_f = self.conv_feature(x)

        return conv_f


origin_image = cv2.imread('./test_image/2.png', -1)  # BGR

img = copy.deepcopy(origin_image)
x = preprocess_input(img)

net = VaModel(vgg)

feature = net(x)
feature = feature.squeeze()
feature = feature.cpu().detach().numpy()

channel = feature.shape[0]
size = feature.shape[1]

image_per_row = 16

rows = channel // image_per_row

display_grad = np.zeros((size * rows, image_per_row * size))

for row in range(rows):
    for col in range(image_per_row):
        channel_image = feature[row * image_per_row + col, :, :]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()

        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')

        display_grad[row*size : (row+1)*size, col*size: (col+1)*size] = channel_image

scale = 1. / size
plt.figure(figsize=(scale * display_grad.shape[1], scale * display_grad.shape[0]))
plt.title('feature')
plt.grid(False)

plt.imshow(display_grad, aspect='auto', cmap='viridis')
plt.show()









