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
dl = DataLoader(ds, batch_size=1)
device = torch.device('cpu')
model = UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/multi/checkpoint_1_110.pt')
# chkpt= torch.load('chkps/data/multi/v1/checkpoint_1_40.pt')
# chkpt= torch.load('chkps/data/multi/v2/checkpoint_60.pt')
chkpt= torch.load('chkps/data/multi/v3/checkpoint_10.pt')

model.load_state_dict(chkpt['model_state_dict'])
model=model.to(device)
model.eval()

diff_list=[]
diff_list2=[]
idx=0

para_frequenc=1
k=2.0
with torch.no_grad():
    for x, y,z in dl:
        
        x,y,z=ds[699]
        x, y,z= x.unsqueeze(0).to(device), y.unsqueeze(0).to(device),torch.tensor(z).unsqueeze(0).to(device)
        x, y,real_num= x.to(device), y.to(device),z.to(device)
        x=(x-x.min())/(x.max()-x.min())
        pred = model(x)

        # pred=pred.squeeze().numpy()
        # if idx%para_frequenc==0:
        #     max_len=0
        #     for t_k in np.arange(0.1,5.0,0.1):
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
        thresh=pred.mean()+k*(pred.std())

        # thresh=pred.mean()+2.0*(pred.std())
        mask=pred.squeeze()>thresh
        mask=mask.numpy()

   

        
        # 标记mask中不同的连通区域
        label_mask = measure.label(mask) 
        all_labels = measure.label(mask)

        all_nums = len(measure.regionprops(all_labels))
        
        
        areas = [r.area for r in measure.regionprops(all_labels)]

        mean_area =  np.median(areas)

        
        more=0

        for region in measure.regionprops(all_labels):
            if region.area > 2*mean_area:
                all_nums+=region.area//mean_area-1
                more+=region.area//mean_area-1
        # for region in measure.regionprops(all_labels):
        #     if region.area < 0.4*mean_area:
        #         # all_labels.pop(region.label)
        #         all_labels[all_labels==region.label]=0
                # all_labels[all_labels==region.label]=0

        label_mask=all_labels


        

        # 计算每个区域的坐标中心点   
        regions = measure.regionprops(label_mask)
        centroids = [region.centroid for region in regions]
 
        
        diff=np.abs(real_num.item()-len(centroids))
        diff_list.append(diff.item())
        diff_list2.append(np.abs(real_num.item()-all_nums))
        print(f"min:{np.min(areas)},max:{np.max(areas)},mean:{np.mean(areas)},median:{np.median(areas)}")

        print(f'id:{idx}, diff:{diff.item()},real:{real_num.item()},len:{len(centroids)},pred:{all_nums},more:{more}')
        # diff=(len(centroids)-real_num).abs()
        # diff_list.append(diff.item())
        # print(f'id:{idx}, diff:{diff.item()},real:{real_num.item()},pred:{len(centroids)}')

        if idx==0:
            fig, ax = plt.subplots(1, 3, figsize=(8, 4))
            ax[0].imshow(x.squeeze())
            ax[0].set_title('input')
            ax[1].imshow(pred.squeeze())
            ax[1].set_title('prediction')
            ax[2].imshow(mask)
            plt.scatter(x=[p[1] for p in centroids], y=[p[0] for p in centroids], c='r', s=5)
            plt.show()

        idx+=1
        


# 汇总结果    
print("Max diff:", max(diff_list))
print("Mean diff:", sum(diff_list)/len(diff_list))

print("Max diff2:", max(diff_list2))
print("Mean diff2:", sum(diff_list2)/len(diff_list2))