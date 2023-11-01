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
from datasets import MultiDatasetV1 as MultiDataset
from torchvision.transforms import functional as F
import random

# data_path = './data/2,1000,500/'
data_path = './data/multi/'
suf="Unet"
bin_gt=False
# bin_gt=True
batch_size=40
trans_rate=0.5
device = torch.device('cuda:0')
# 自定义Dataset加载TIFF数据
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
        
# train_ds=RandomDataset(data_path,split='train')
# train_image,train_gt=train_ds[0]
# print(train_image.shape,train_gt.shape)

def transform_image( image, gt):
    # 随机选择一种变换


    transform_type = random.choice(['rotate', 'bigger','smaller', 'fill_zero'])
    if transform_type == 'rotate':
        angle = random.uniform(-180, 180)
        image[:] = F.rotate(image, angle,fill=0)
        gt[:] = F.rotate(gt, angle,fill=0)
    elif transform_type == 'bigger':
        scale_factor = random.uniform(1.0, 2.0)
        output_size = (int(128 * scale_factor), int(128 * scale_factor))
        if random.random() < 0.5:
            image = F.resize(image, output_size,antialias=True)
            image=image[:,:, image.shape[2]-128:image.shape[2],image.shape[3]-128:image.shape[3]]
            gt = F.resize(gt, output_size,antialias=True)
            gt=gt[:,:, gt.shape[2]-128:gt.shape[2],gt.shape[3]-128:gt.shape[3]]
       
        else:
            image = F.resize(image, output_size,antialias=True)[:,:,:128,:128]
            gt = F.resize(gt, output_size,antialias=True)[:,:,:128,:128]
    elif transform_type == 'smaller':
        scale_factor = random.uniform(0.85, 1.0)
        output_size = (int(128 * scale_factor), int(128 * scale_factor))
        new_image = F.resize(image, output_size,antialias=True)
        new_gt = F.resize(gt, output_size,antialias=True)
        image[:,:,:,:]=0
        gt[:,:,:,:]=0
        if random.random() < 0.5:
            image[:,:,:int(128*scale_factor),:int(128*scale_factor)] = new_image
            gt[:,:,:int(128*scale_factor),:int(128*scale_factor)] = new_gt
        else:
            image[:,:,128-(new_image.shape[2]):128,128-(new_image.shape[3]):128] = new_image
            gt[:,:,128-(new_image.shape[2]):128,128-(new_image.shape[3]):128] = new_gt
        # scale_factor = random.uniform(0.5, 2.0)
        # # 计算填充的大小
        # pad_width = int((scale_factor - 1) * image.size()[2] / 2)
        # pad_height = int((scale_factor - 1) * image.size()[3] / 2)
        # # 进行缩放和填充
        # image = F.affine(image, angle=0, translate=(0, 0), scale=scale_factor, shear=0)
        # gt = F.affine(gt, angle=0, translate=(0, 0), scale=scale_factor, shear=0)
        # image = F.pad(image, padding=(pad_width, pad_height, pad_width, pad_height), fill=0)
        # gt = F.pad(gt, padding=(pad_width, pad_height, pad_width, pad_height), fill=0)
    elif transform_type == 'fill_zero':
        x_size = image.size()
        gt_size = gt.size()

        x_width, x_height = x_size[2], x_size[3]
        gt_width, gt_height = gt_size[2], gt_size[3]

        pad_x = random.randint(0, x_width)
        pad_y = random.randint(0, x_height)
        pad_w = random.randint(0, x_width - pad_x)
        pad_h = random.randint(0, x_height - pad_y)

        # 进行填充操作
        image[:, :, pad_x:pad_x+pad_w, pad_y:pad_y+pad_h] = 0
        gt[:, :, pad_x:pad_x+pad_w, pad_y:pad_y+pad_h] = 0
    

 
    
    return image[:,:,:128,:128], gt[:,:,:128,:128]

# 训练函数
def train(model, train_loader, val_loader, num_epochs):
    device = torch.device('cuda:0') # 使用GPU
    model.to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.AdamW(model.parameters())
    best_val_loss = float('inf') 
    patience = 20
    patience_counter = 0
    train_losses, val_losses = [], []
 
    epoch_id=0
    for epoch in range(num_epochs):
        idx=0
        model.train()
        for x, y in train_loader:
            # 将像素值缩放到[0, 1]范围
            # max_value=x.max()
            # x = x / max_value
            # if(idx==0)  :
            #     plt.imshow(x[0].permute(1,2,0))
            #     plt.show()

            x ,y = x.to(device), y.to(device)

            max_value=x.max()
            # print(f'max_value:{max_value}')
            # x = (x-x.mean()) / (x.std()+1e-8)
            x[:]=(x-x.min())/(x.max()-x.min())

            # y=(y-y.mean())/(y.std()+1e-8)
            y[:]=(y-y.min())/(y.max()-y.min())
            # y=y/y.max()
            if random.random() <trans_rate:
                x[:],y[:] = transform_image(x,y)
            pred= model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx==0 and epoch_id==0:
                print(f'x_min:{x.min()},x_max:{x.max()},y_min:{y.min()},y_max:{y.max()},pred_min:{pred.min()},pred_max:{pred.max()}')
                # fig, ax = plt.subplots(1, 3, figsize=(8, 4))
                # ax[0].imshow(x[0].permute(1,2,0).cpu().detach().numpy())
                # ax[1].imshow(y[0].permute(1,2,0).cpu().detach().numpy())
                # ax[2].imshow(pred[0].permute(1,2,0).cpu().detach().numpy())
                # plt.show()
            idx+=1
            
        # 验证及保存模型
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, f'chkps/{data_path}{suf}/checkpoint_latest.pt')
            # 验证loss

        val_loss = 0 
        model.eval()
        with torch.no_grad():
            t_id=0
            for x, y in val_loader:
                
                x, y = x.to(device), y.to(device)
                max_value=x.max()
            # print(f'max_value:{max_value}')
                # x = x / max_value
                # x = (x-x.mean()) / (x.std()+1e-8)
                x=(x-x.min())/(x.max()-x.min())
                # y=y/y.max() 
                # y=(y-y.mean())/(y.std()+1e-8)
                y=(y-y.min())/(y.max()-y.min())
                if random.random() < trans_rate:
                    x,y = transform_image(x,y)
                pred = model(x)
                # loss = criterion(pred, y)
                val_loss += criterion(pred, y).item()
                fig,ax=plt.subplots(1,3)
                ax[0].imshow(x[0].permute(1,2,0).cpu().detach().numpy())
                ax[1].imshow(y[0].permute(1,2,0).cpu().detach().numpy())
                ax[2].imshow(pred[0].permute(1,2,0).cpu().detach().numpy())
                plt.savefig(f'chkps/{data_path}{suf}/log/val_{epoch_id+1}.jpg')
                plt.cla()
                plt.clf()
                plt.close()
                t_id+=1
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            torch.save({
    'epoch': epoch+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, f'chkps/{data_path}{suf}/checkpoint_best.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print('Early stopping.')
            break
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, Patience: {patience_counter}/{patience}') 
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        epoch_id+=1
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'chkps/{data_path}{suf}/log/loss.jpg')
    # plt.show()   
# 准备数据        
# train_ds = RandomDataset(data_path,split='train')  
# train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
# val_ds = RandomDataset(data_path,split='val')
# val_dl = DataLoader(val_ds, batch_size=64)

train_ds = MultiDataset(split='train',bin_gt=bin_gt)
# train_ds = MultiDataset(split='train',bin_gt=False)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_ds = MultiDataset(split='val',bin_gt=bin_gt)
# val_ds = MultiDataset(split='val',bin_gt=False)
val_dl = DataLoader(val_ds, batch_size=batch_size)

model = UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/multi/v5/checkpoint_30.pt')
# model.load_state_dict(chkpt['model_state_dict'])
model.to(device)
# model.train()
# checkpoint = torch.load('chkps/data/2,1000/checkpoint60.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
if not os.path.exists(f'chkps/{data_path}{suf}'):
  os.makedirs(f'chkps/{data_path}{suf}')
if not os.path.exists(f'chkps/{data_path}{suf}/log'):
  os.makedirs(f'chkps/{data_path}{suf}/log')
train(model, train_dl, val_dl, 20000)


    

