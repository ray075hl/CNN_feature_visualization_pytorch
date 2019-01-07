import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def Show_origin_and_output(origin, I):
   """
   Show final result.
   """
   plt.figure(figsize=(12, 6))
   plt.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=0.005, hspace=0)

   plt.subplot(121),plt.imshow(np.flip(origin, 2)),plt.title('Origin')
   plt.axis('off')
   plt.subplot(122),plt.imshow(np.flip(I, 2)),plt.title('Mixed Heatmap')
   plt.axis('off')
   plt.savefig('result1.jpg', bbox_inches='tight', pad_inches=0)
   plt.show()


def preprocess_input(img):
    image = 1.0 * img/255.0
    image = np.flip(image, 2)  # BGR -> RGB

    image = cv2.resize(image, (224, 224))  # vgg16 input size: 224
    # Mean and Std of Imagenet dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image[:, :, 0] -= mean[0]
    image[:, :, 1] -= mean[1]
    image[:, :, 2] -= mean[2]

    image[:, :, 0] /= std[0]
    image[:, :, 1] /= std[1]
    image[:, :, 2] /= std[2]

    image = np.transpose(image, (2, 0, 1))  # channel first
    image = image[np.newaxis, ...]  # 4D input for network

    image = torch.FloatTensor(image)  # convert to tensor

    return image


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    x = x.transpose(1, 2, 0)
    return x
