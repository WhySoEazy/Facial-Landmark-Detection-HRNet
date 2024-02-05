import torch
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
import lib.models as models
from lib.utils import utils
import matplotlib.pyplot as plt
from lib.config import config, update_config

import os
data = torch.load("./output/WFLW/face_alignment_wflw_hrnet_w18/predictions.pth")
print(data.size())
# print(data[0])
# print(sum([len(os.listdir(f"./data/wflw/images/{i}")) for i in os.listdir("./data/wflw/images")]))


img = cv2.imread("C:/Users/Asus/Desktop/PartA_01803.jpg")
new_img = img.copy()
for idx in range(8, 9):
    for i in data[idx]:
        # print(i)
        cv2.circle(new_img, i.long().numpy(), 5, (255, 255, 255), 3) 

plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img)

# plt.subplot(1, 3, 2)
# plt.scatter(data[0, :, 0], data[0, :, 1], s=10, marker='.', c='r')
plt.subplot(1, 3, 3)
plt.title("output")
plt.imshow(new_img)

plt.show()