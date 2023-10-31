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
from PIL import Image
from PIL import ImageEnhance
data_path = './data/'
# 自定义Dataset加载TIFF数据
class PreDataset(Dataset):
    def __init__(self, name,split='train',contrast = 1):
        self.frames = self.tiff_to_array(data_path+name)
        self.split=split
        self.contrast = contrast

    def __getitem__(self, idx):
        image = self.frames[idx]
        image=image[:128,:128,:]
        # print(image.shape)

        image=Image.fromarray(np.uint8(image[:,:,0]*255))
        enh_con = ImageEnhance.Contrast(image)
        contrast = 9
        image= enh_con.enhance(self.contrast)
        image = np.asarray(image)
        image=np.expand_dims(image,axis=-1)
        # plt.imshow(image)
        # plt.show()
        return torch.tensor(image).permute(2,0,1).float()
       
    
    def __len__(self):
          return len(self.frames)
    
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
    
class RandomDataset(Dataset):
    def __init__(self, data_path,split='train',ret_num=False,bin_gt=False):
        self.frames = self.tiff_to_array(data_path+'Artificial dataset.tif')
        self.split=split
        self.data_path=data_path
        self.ret_num=ret_num
        self.nums=[]
        self.bin_gt=bin_gt
        if self.bin_gt:
            device = torch.device('cuda:0') 
            model = UNet(n_channels=1, n_classes=1)
            # chkpt= torch.load('chkps/data/2,1000,500/checkpoint_1_240.pt') 
            # chkpt= torch.load('chkps/data/2,1000/checkpoint60.pt')
            # chkpt= torch.load('chkps/data/multi/checkpoint_1_70.pt') 
            # chkpt= torch.load('chkps/data/multi/checkpoint_1_110.pt') 
            # chkpt= torch.load('chkps/data/multi/v1/checkpoint_1_40.pt') 
            # chkpt= torch.load('chkps/data/multi/v2/checkpoint_60.pt') 
            chkpt= torch.load('chkps/data/multi/v3/checkpoint_20.pt')

            model.load_state_dict(chkpt['model_state_dict'])
            model=model.to(device)
            model.eval()
            self.model=model

        if ret_num:
            ids=[]
            df = pd.read_csv(self.data_path+"frame_logger.csv") 
            grouped = df.groupby('frame')

            # 遍历每个frame
            diff_list = []
            nums_dict={}
            for frame, group in grouped:
                ids.append(frame-1)
                # 真实点数量
                real_num = len(group)
                # self.nums.append(real_num)
                nums_dict[frame-1]=real_num
            for i in range(1000):
                if not i in ids:
                    nums_dict[i]=0
            for i in range(1000):
                self.nums.append(nums_dict[i])

                
                


    
    def __getitem__(self, idx):
        if self.split=='val':
            idx+=900
        image = self.frames[idx]
        
        if self.bin_gt:
            input_name=self.data_path+'ground_truth/frame_{}.jpg'.format(idx)
            gt_name=self.data_path+'ground_truth_bin/frame{}.jpg'.format(idx)
        else:
            gt_name=self.data_path+'ground_truth/frame_{}.jpg'.format(idx)
        # print(gt_name)
        gt =cv2.imread(gt_name,0)
        if self.bin_gt:
            # x=cv2.imread(input_name,0)
            x=image
            x=torch.tensor(x).unsqueeze(0).unsqueeze(0).squeeze(-1).float().cuda()
            # print(x.shape)
            y=self.model(x)
            # print(x.shape,y.shape)
            image=y.squeeze(0).squeeze(0).unsqueeze(-1).cpu().detach().numpy()
            # fig,ax=plt.subplots(1,3)
            # ax[0].imshow(gt)
            # ax[1].imshow(image)
            # ax[2].imshow(x.squeeze(0).squeeze(0).cpu().detach().numpy())
            # plt.show()
            # print(image.shape)
            # print(gt.shape)
            # plt.imshow(gt)
            # plt.show()
        if not self.ret_num:
            return torch.tensor(image).permute(2,0,1).float(), torch.tensor(gt).unsqueeze(-1).permute(2,0,1).float()
        else:
            return torch.tensor(image).permute(2,0,1).float(), torch.tensor(gt).unsqueeze(-1).permute(2,0,1).float(),self.nums[idx]  
    
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
   
data_path1 = './data/2,1000/'
data_path2 = './data/2,1000,500/'
data_path3='./data/0.05,1600/'
data_path4='./data/0.05/'
data_path5='./data/0.1/'
data_path6='./data/0.15/'
data_path7='./data/0.2/'
data_path_epl='./data/epl/'
data_path_cell='./data/cell/'
data_path_gfp='./data/gfp/'

class MultiDataset(Dataset):
    def __init__(self, split='train',ret_num=False,bin_gt=False):
        self.data1=RandomDataset(data_path1,split,ret_num,bin_gt=bin_gt)
        self.data2=RandomDataset(data_path2,split,ret_num,bin_gt=bin_gt)
        self.data3=RandomDataset(data_path3,split,ret_num,bin_gt=bin_gt)
        self.data4=RandomDataset(data_path4,split,ret_num,bin_gt=bin_gt)
        self.data_epl=RandomDataset(data_path_epl,split,ret_num,bin_gt=bin_gt)
        self.data_cell=RandomDataset(data_path_cell,split,ret_num,bin_gt=bin_gt)
        self.data_gfp=RandomDataset(data_path_gfp,split,ret_num,bin_gt=bin_gt)
        # self.data1=self.data_gfp
        self.split=split
    def __len__(self):
        if self.split=='train':
            return len(self.data1)+len(self.data2)+len(self.data3)+len(self.data4)+len(self.data_epl)+len(self.data_cell)+len(self.data_gfp)-700
        else:
            return 700
    def __getitem__(self, idx):
        if self.split=='val':
            if idx<100:
                return self.data1[idx]
            elif idx<200:
                return self.data2[idx-100]
            elif idx<300:
                return self.data_epl[idx-200]
            elif idx<400:
                return self.data_cell[idx-300]
            elif idx<500:
                return self.data_gfp[idx-400]
            elif idx<600:
                return self.data3[idx-500]
            else:
                return self.data4[idx-600]
            
            
        if idx<900:
            return self.data1[idx]
        elif idx<1800:
            return self.data2[idx-900]
        elif idx<2700:
            return self.data_epl[idx-1800]
        elif idx<3600:
            return self.data_cell[idx-2700]
        elif idx<4500:
            return self.data_gfp[idx-3600]
        elif idx<5400:
            return self.data3[idx-4500]
        else:
            return self.data4[idx-5400]
        
class MultiDatasetV1(Dataset):
    def __init__(self, split='train',ret_num=False,bin_gt=False):
        self.datasets =  [
            RandomDataset(data_path1, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path2, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path3, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path4, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path5, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path6, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path7, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path_epl, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path_cell, split, ret_num, bin_gt=bin_gt),
            RandomDataset(data_path_gfp, split, ret_num, bin_gt=bin_gt)
        ]
        
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx):
        cumulative_length = 0
        for dataset in self.datasets:
            if idx < cumulative_length + len(dataset):
                return dataset[idx - cumulative_length]
            cumulative_length += len(dataset)

        raise IndexError('index out of range')
 
