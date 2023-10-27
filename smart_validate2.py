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
import skimage.measure as measure


ds=MultiDataset(split='val',ret_num=True)
dl = DataLoader(ds, batch_size=1,shuffle=True)
device = torch.device('cpu')
model = UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/multi/checkpoint_1_110.pt')

chkpt= torch.load('chkps/data/multi/v2/checkpoint_60.pt')
model.load_state_dict(chkpt['model_state_dict'])
model=model.to(device)
model.eval()
print("model loaded")
# model2=UNet(n_channels=1, n_classes=1,bilinear=True)
model2=UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/multi/_bin/checkpoint_1_220.pt')
chkpt= torch.load('chkps/data/multi/v2_bin/checkpoint_10.pt')

# print(chkpt.keys())
model2.load_state_dict(chkpt['model_state_dict'])
model2=model2.to(device)
model2.eval()
print("model2 loaded")
diff_list=[]
idx=0

para_frequenc=1
k=5.0
with torch.no_grad():
    for x, y,z in dl:
        x, y,real_num= x.to(device), y.to(device),z.to(device)
        pred1 = model(x)
        print(x.min(),x.max())
        # pred=pred/pred.max()
        print(pred1.min(),pred1.max())
        pred=model2(pred1)
        print(pred.shape)
        # print(f"sum:{pred.sum()}")
        # print(f"num:{(pred>pred.mean()).sum()}")
        # pred=pred.squeeze().numpy()
        # if idx%para_frequenc==0:
        #     max_len=0
        #     for t_k in np.arange(1,5.0,0.1):
        #         t_thresh=pred.mean()+t_k*(pred.std())
        #         t_mask=pred.squeeze()>t_thresh
        #         t_mask=t_mask.numpy()
        #         t_label_mask = measure.label(t_mask)
        #         t_all_labels = measure.label(t_mask)
        #         t_regions = measure.regionprops(t_label_mask)
        #         t_centroids = [region.centroid for region in t_regions]
        #         t_len=len(t_centroids)
        #         if t_len>max_len:
        #             k=t_k
        #             max_len=t_len
        #     print(f'idx:{idx},k:{k}')
        k=1.0
        thresh=pred.mean()+k*(pred.std())

        # thresh=pred.mean()+2.0*(pred.std())
        mask=pred.squeeze()>thresh
        mask=mask.numpy()

   

        
        # 标记mask中不同的连通区域
        label_mask = measure.label(mask) 
        all_labels = measure.label(mask)

        all_nums = len(measure.regionprops(all_labels))
        
        
        areas = [r.area for r in measure.regionprops(all_labels)]

 

        # 计算每个区域的坐标中心点   
        regions = measure.regionprops(label_mask)
        centroids = [region.centroid for region in regions]
 
        
        diff=(all_nums-real_num).abs()
        diff_list.append(diff.item())
        print(f"min:{np.min(areas)},max:{np.max(areas)},mean:{np.mean(areas)},median:{np.median(areas)}")

        print(f'id:{idx}, diff:{diff.item()},real:{real_num.item()},len:{len(centroids)}')
        # diff=(len(centroids)-real_num).abs()
        # diff_list.append(diff.item())
        # print(f'id:{idx}, diff:{diff.item()},real:{real_num.item()},pred:{len(centroids)}')

        if idx==0:
            fig, ax = plt.subplots(1, 4, figsize=(8, 4))
            ax[0].imshow(x.squeeze())
            ax[0].set_title('input')
            ax[1].imshow(pred1.squeeze())
            ax[1].set_title('prediction1')
            ax[2].imshow(pred.squeeze())
            ax[2].set_title('prediction')
            ax[3].imshow(mask)
            plt.scatter(x=[p[1] for p in centroids], y=[p[0] for p in centroids], c='r', s=5)
            plt.show()

        idx+=1
        


# 汇总结果    
print("Max diff:", max(diff_list))
print("Mean diff:", sum(diff_list)/len(diff_list))