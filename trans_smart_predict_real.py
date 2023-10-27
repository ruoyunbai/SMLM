import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tifffile 
from torchvision import transforms
from skimage import io

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from datasets import PreDataset
import skimage.measure as measure
from trans_unet.transunet import TransUNet as UNet


val_ds = PreDataset("cell8_2_10000.tif",split='val',contrast=10)

# val_ds = PreDataset("img_000000005_Default_000.tif",split='val',contrast=1)
# val_ds = PreDataset("EPFL_Microtubule_Alexa647.tif",split='val')
val_dl = DataLoader(val_ds, batch_size=1,shuffle=False)
device = torch.device('cpu') 
model = UNet(img_dim=128,
                          in_channels=1,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)

chkpt= torch.load('chkps/data/multi/v6(TransUnet)/checkpoint_20.pt')
model.load_state_dict(chkpt['model_state_dict'])
model=model.to(device)
model.eval()


with torch.no_grad():
    x = next(iter(val_dl))[:,:128,:128]
    x = x.to(device)
    x=(x-x.min())/(x.max()-x.min())
    # 中值滤波去噪
    origin_x=x
    print(x.shape)
    # x = cv2.medianBlur(x.squeeze(0).squeeze(0).numpy(), 5)

    # 高斯滤波进一步去噪
    # x= torch.tensor(cv2.GaussianBlur(x.squeeze(0).squeeze(0).numpy(), (5, 5), 0) ).unsqueeze(0).unsqueeze(0)
    pred = model(x)


nums=[]
xs=[]
subs=[]
last=0
# for k in np.arange(0.1,5.0,0.1):
if True:
    k=3.0
    fig, ax = plt.subplots(1, 5, figsize=(12, 4))
    ax[0].imshow(origin_x.squeeze())
    ax[0].set_title('input')
    ax[1].imshow(x.squeeze())
    ax[1].set_title('input')
    ax[2].imshow(pred.squeeze())
    ax[2].set_title('prediction')
    # print(pred.max())
    # print(pred.min())
    # print(pred.mean())    

    threshold=pred.mean()+k*(pred.std())

    # 将预测图像转换为灰度图像
    # print(pred.squeeze(0).squeeze(0).shape)
    # gray_img = pred.squeeze(0).squeeze(0).numpy().astype(np.uint8)
    # ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print(ret,thresh)
    # 自适应阈值算法
    # thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -2)
    # ax[4].imshow(thresh)
    # mask=pred.squeeze()>pred.mean()
    # mask1= torch.sigmoid(pred)>0.5
    # ax[1].imshow(mask1.squeeze())
    # ax[1].set_title('mask1')
    # print(pred.sigmoid().sum())
    mask=pred.squeeze()>threshold
    mask=mask.numpy()


    # 标记mask中不同的连通区域
    all_labels = measure.label(mask)


    # 计算每个区域的坐标中心点   
    regions = measure.regionprops(all_labels)
    centroids = [region.centroid for region in regions]
    # print(centroids)
    # print(len(centroids))


    
    all_nums = len(centroids)
    more=0

    areas = [r.area for r in measure.regionprops(all_labels)]
    mean_area=np.median(areas)
    max_area=np.max(areas)

    areas=np.array([])
    for region in measure.regionprops(all_labels):
        areas=np.append(areas,region.area)
        if region.area > 2*mean_area:
            all_nums+=region.area//mean_area-1
            more+=region.area//mean_area-1

    # for region in measure.regionprops(all_labels):
    #     if region.area < 0.4*mean_area:
    #         # all_labels.pop(region.label)
    #         all_labels[all_labels==region.label]=0

    # for region in measure.regionprops(all_labels):
    #     if region.area <max_area/10:
    #         # all_labels.pop(region.label)
    #         all_labels[all_labels==region.label]=0
    
    regions = measure.regionprops(all_labels)
    centroids = [region.centroid for region in regions]
    label_mask=all_labels
        

    print(f"min:{np.min(areas)},max:{np.max(areas)},mean:{np.mean(areas)},median:{np.median(areas)}")
    ax[3].imshow(mask)
    # 用红点标记所有检测到的亮点坐标 
    ax[0].scatter(x=[p[1] for p in centroids], y=[p[0] for p in centroids], c='r', s=2)
    
    ax[3].scatter(x=[p[1] for p in centroids], y=[p[0] for p in centroids], c='r', s=2)
    print(f"threshold:{threshold},k:{k},len:{len(centroids)},all_nums:{all_nums},more:{more},s_std:{areas.std()},s_mean:{areas.mean()}")
    # print(len(centroids))
    print(pred.sum())

    # xs.append(k)
    # nums.append(len(centroids))
    # ax[4].plot(xs,nums)
    # subs.append(len(centroids)-last)
    # last=len(centroids)
    # ax[4].plot(xs,subs)
    # ax[4].imshow(pred2.squeeze())

    plt.show()

