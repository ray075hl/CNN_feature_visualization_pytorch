"""
Pytorch implement of Grad-CAM.

Grad-CAM: visual explanations from deep networks
via gradient-based localization. (https://arxiv.org/abs/1610.02391)

CAM: Class Activation Map
"""

import sys
import requests

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

import numpy as np
import cv2

import matplotlib.pyplot as plt
import copy

from utils import Show_origin_and_output, preprocess_input

DEBUG = True
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# Let's get our class labels.
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}


class CamModel(nn.Module):
    def __init__(self, pretrained_model):
        super(CamModel, self).__init__()
        self.feature_model = torch.nn.Sequential(*list(pretrained_model.features.children())[:-2])  # -2 is last conv layer
        self.cnn_lasttwo = torch.nn.Sequential(*list(pretrained_model.features.children())[-2:])
        self.classifier = torch.nn.Sequential(*list(pretrained_model.classifier.children())[:])

    def forward(self, x):
        batch = x.size()[0]
        last_conv_f = self.feature_model(x)
        y = self.cnn_lasttwo(last_conv_f)
        y = y.view(batch, -1)
        out = self.classifier(y)

        return last_conv_f, out


# https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94/6
grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


if __name__ == '__main__':
    img_path = sys.argv[1]

    if len(sys.argv) < 2:
        print("Usage: python main.py [image path]")
        exit()

    origin_image = cv2.imread(img_path, -1)  # BGR
    img = copy.deepcopy(origin_image)

    x = preprocess_input(img)  # preprocess for vgg16

    model = vgg16(pretrained=True)
    net = CamModel(model)

    last_conv_output, class_output = net(x)
    print(labels[torch.argmax(class_output).item()])
    net.zero_grad()

    class_output = class_output[:, torch.argmax(class_output).item()]

    last_conv_output.register_hook(save_grad('last_conv_output'))
    class_output.backward()

    pooled_grads = grads['last_conv_output']

    # Define the important weights by gradient's mean of each feature channel respect to class output.
    # 用类别相对于通道的梯度对这个特征图中的每个通道进行加权
    pooled_grads = torch.mean(pooled_grads, dim=(0,2,3))

    last_conv_output = last_conv_output.squeeze()

    # last_conv_output.size()[0] is feature channel.
    for i in range(last_conv_output.size()[0]):
        last_conv_output[i, :, :] *= pooled_grads[i]

    last_conv_output_np = last_conv_output.cpu().detach().numpy()

    heatmap = np.mean(last_conv_output_np, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    if DEBUG:
        plt.matshow(heatmap)

    heatmap = cv2.resize(heatmap, (origin_image.shape[1], origin_image.shape[0]))
    heatmap = np.uint8(heatmap*255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + origin_image

    Show_origin_and_output(origin_image, superimposed_img/255.0)

