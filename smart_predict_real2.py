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
from datasets import PreDataset
import skimage.measure as measure


# val_ds = PreDataset("cell8_2_10000.tif",split='val',contrast=10)

val_ds = PreDataset("img_000000005_Default_000.tif",split='val',contrast=1)
# val_ds = PreDataset("EPFL_Microtubule_Alexa647.tif",split='val')
val_dl = DataLoader(val_ds, batch_size=1,shuffle=False)
device = torch.device('cpu') 
model = UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/2,1000,500/checkpoint_1_240.pt') 
# chkpt= torch.load('chkps/data/2,1000/checkpoint60.pt')
# chkpt= torch.load('chkps/data/multi/checkpoint_1_70.pt') 
# chkpt= torch.load('chkps/data/multi/checkpoint_1_110.pt') 
# chkpt= torch.load('chkps/data/multi/v1/checkpoint_1_160.pt')
# chkpt= torch.load('chkps/data/multi/v2/checkpoint_20.pt')
chkpt= torch.load('chkps/data/multi/v3/checkpoint_20.pt')
model.load_state_dict(chkpt['model_state_dict'])
model=model.to(device)
model.eval()

# model2=UNet(n_channels=1, n_classes=1,bilinear=True)
# chkpt= torch.load('chkps/data/multi/_bin/checkpoint_1_220.pt')

model2=UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/multi/_bin/checkpoint_1_220.pt')
# chkpt= torch.load('chkps/data/multi/v2_bin/checkpoint_90.pt')
chkpt= torch.load('chkps/data/multi/v3_bin/checkpoint_20.pt')
# print(chkpt.keys())
model2.load_state_dict(chkpt['model_state_dict'])
model2=model2.to(device)
model2.eval()

with torch.no_grad():
    x = next(iter(val_dl))[:,:128,:128]
    print(x.min(),x.max())
    x = x.to(device)
    x=(x-x.min())/(x.max()-x.min())
    pred1 = model(x)
    pred1=(pred1-pred1.min())/(pred1.max()-pred1.min())
    print(pred1.min(),pred1.max())
    # pred=pred/pred.max()
    print(pred1.min())

    pred=model2(pred1)



# for k in np.arange(1.0,5.0,0.1):
k=4.0
fig, ax = plt.subplots(1, 4, figsize=(8, 4))

ax[1].imshow(pred1.squeeze())
ax[1].set_title('prediction1')
ax[2].imshow(pred.squeeze())
ax[2].set_title('prediction')
# print(pred.max())
# print(pred.min())
# print(pred.mean())    
threshold=pred.mean()+k*(pred.std())
# mask=pred.squeeze()>pred.mean()
mask=pred.squeeze()>threshold
mask=mask.numpy()


# 标记mask中不同的连通区域
label_mask = measure.label(mask) 
all_labels = measure.label(mask)


# 计算每个区域的坐标中心点   
regions = measure.regionprops(label_mask)
centroids = [region.centroid for region in regions]
# print(centroids)
# print(len(centroids))



all_nums = len(centroids)
more=0

areas = [r.area for r in measure.regionprops(all_labels)]
mean_area=np.median(areas)

for region in measure.regionprops(all_labels):
    if region.area > 2*mean_area:
        all_nums+=region.area//mean_area-1
        more+=region.area//mean_area-1
print(f"min:{np.min(areas)},max:{np.max(areas)},mean:{np.mean(areas)},median:{np.median(areas)}")

ax[0].imshow(x.squeeze())
ax[0].set_title('input')

ax[0].scatter(x=[p[1] for p in centroids], y=[p[0] for p in centroids], c='r', s=2)


ax[3].imshow(mask)
# 用红点标记所有检测到的亮点坐标 
plt.scatter(x=[p[1] for p in centroids], y=[p[0] for p in centroids], c='r', s=5)
print(f"threshold:{threshold},k:{k},len:{len(centroids)},all_nums:{all_nums},more:{more}")
print(len(centroids))
print(pred.sum())
plt.show()

