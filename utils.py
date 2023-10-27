import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tifffile 
from torchvision import transforms
from skimage import io
from unet import UNet
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from datasets import MultiDataset

import cv2
import numpy as np

class pointFinder():
    def __init__(self,img):
        # 读取图像并转化为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 图像二值化
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 

        # 寻找轮廓 
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历每个轮廓,计算其矩以得到中心点
        centroid_list = []
        for cnt in contours:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroid_list.append((cx, cy))

        # 显示结果
        print(centroid_list)