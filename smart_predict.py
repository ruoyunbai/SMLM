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
# data_path = './data/2,1000,500/'
data_path = './data/2,1000/'

class RandomDataset(Dataset):
    def __init__(self, data_path,split='train'):
        self.frames = self.tiff_to_array(data_path+'Artificial dataset.tif')
        self.split=split
    
    def __getitem__(self, idx):
        if self.split=='val':
            idx+=100
        image = self.frames[idx]
        gt_name=data_path+'ground_truth/frame_{}.jpg'.format(idx)
        gt =cv2.imread(gt_name,0)
        return torch.tensor(image).permute(2,0,1).float(), torch.tensor(gt).unsqueeze(-1).permute(2,0,1).float()
       
    
    def __len__(self):
        if self.split=='train':
          return len(self.frames)-100
        else:
          return 100
    
    def tiff_to_array(self,input_file, normalize=True):
        """Transforms a tiff image/stack into a (n, x, y, 1) numpy array of floats.

        This function is used to convert tiff images to input arrays. By default,
        the inputs are converted to floats and normalized between 0 and 1 by
        dividing pixel values by the bit depth of the images.

        Parameters
        ----------
        input_file : str
            Path and filename to the TIF image/stack.
        normalize : bool
            Determines whether the images will be normalized.

        Returns
        -------
        data : array_like
            The normalized array of images.

        """
        data = io.imread(input_file)
        if data.ndim == 3:
            data = data[:, :, :, np.newaxis]
        elif data.ndim == 2:
            data = data[np.newaxis, :, :, np.newaxis]
        else:
            raise ImageDimError('Tiff image should be grayscale and 2D '
                                '(3D if stack)')
        # Converting from uint to float
        if data.dtype == 'uint8':
            max_uint = 255
        elif data.dtype == 'uint16':
            max_uint = 2 ** 16 - 1
        else:
            raise ImageTypeError('Tiff image type should be uint8 or uint16')
        data = data.astype('float')

        if normalize:
            data /= max_uint

        return data
    
from datasets import MultiDataset


ds=MultiDataset(split='val',ret_num=True)
# val_ds = RandomDataset(data_path,split='val')
val_ds=ds
val_dl = DataLoader(val_ds, batch_size=1,shuffle=False)
device = torch.device('cpu') 
model = UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/2,1000,500/checkpoint_1_240.pt') 
chkpt= torch.load('chkps/data/multi/checkpoint_1_110.pt') 
# chkpt= torch.load('chkps/data/2,1000/checkpoint60.pt') 
model.load_state_dict(chkpt['model_state_dict'])
model=model.to(device)
model.eval()


with torch.no_grad():
    x, y,z = next(iter(val_dl))
    x, y,z = x.to(device), y.to(device),z.to(device)
    pred = model(x)

for k in np.arange(1.0,5.0,0.1):
    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(x.squeeze(), cmap='gray')
    ax[0].set_title('input')
    ax[1].imshow(y.squeeze(), cmap='gray')
    ax[1].set_title('ground truth')
    ax[2].imshow(pred.squeeze(), cmap='gray')
    ax[2].set_title('prediction')
    print(pred.sum(),y.sum())
    threshhold=pred.mean()+k*(pred.std())
    mask=pred.squeeze()>threshhold
    mask=mask.numpy()


    import skimage.measure as measure

    # 标记mask中不同的连通区域
    label_mask = measure.label(mask) 
    all_labels = measure.label(mask)

    # 计算每个区域的坐标中心点   
    regions = measure.regionprops(label_mask)
    centroids = [region.centroid for region in regions]


    all_nums = len(centroids)
    more=0

    areas = [r.area for r in measure.regionprops(all_labels)]
    mean_area=np.median(areas)

    for region in measure.regionprops(all_labels):
        if region.area > 2*mean_area:
            all_nums+=region.area//mean_area-1
            more+=region.area//mean_area-1

    print(f"threshhold:{threshhold},k:{k},real:{z.item()},num:{len(centroids)},all_nums:{all_nums},more:{more}")

    ax[3].imshow(mask)
    # print()
    # 用红点标记所有检测到的亮点坐标 
    plt.scatter(x=[p[1] for p in centroids], y=[p[0] for p in centroids], c='r', s=20)

    plt.show()


