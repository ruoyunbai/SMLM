import torch
import argparse

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tifffile
from torchvision import transforms 
from skimage import io, measure
from unet import UNet
from SwinUnet.vision_transformer import SwinUnet 
from TransUnet.vit_seg_modeling import VisionTransformer as TransUNet
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from datasets import MultiDatasetV1 as MultiDataset
def Trans_parse():
    parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=128, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()
parser = argparse.ArgumentParser()
Trans_parse()
args = parser.parse_args()
ds = MultiDataset(split='test', ret_num=True)
dl = DataLoader(ds, batch_size=1)
device = torch.device('cpu')
# model = UNet(n_channels=1, n_classes=1)
# chkpt= torch.load('chkps/data/multi/Unet/checkpoint_best.pt')
# Recall: 0.9443936440020013
# Precision: 0.5689251972186711

# model = SwinUnet(config=None,img_size=128,num_classes=1)
# chkpt= torch.load('chkps/data/multi/SwinUnet/checkpoint_best.pt')
# Recall: 0.9629108704866864
# Precision: 0.22673326618940035


config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
model = TransUNet(config_vit,img_size=128,num_classes=1)
chkpt= torch.load('chkps/data/multi/transUnet/checkpoint_best.pt')
# Recall: 0.9607236072360723
# Precision: 0.5968398452351665

model.load_state_dict(chkpt['model_state_dict'])   
model = model.to(device)
model.eval()

total_tp = 0
total_fp = 0   
total_fn = 0
total_tn=0
id=0
with torch.no_grad():
    for x, y,z in dl:
        x, y, real_num = x.to(device), y.to(device), z.to(device)
        x=(x-x.min())/(x.max()-x.min())
        pred = model(x)
        
        pred_seg = pred.squeeze() > pred.squeeze().mean() + pred.squeeze().std()
        y_seg = y .squeeze()> y.squeeze().mean() + y.squeeze().std()

        # 获取预测和 ground truth 的连通区域
        pred_labels = measure.label(pred_seg.cpu().numpy())
        y_labels = measure.label(y_seg.cpu().numpy())

        # print(y_labels.shape)
        y_centroids = [region.centroid for region in measure.regionprops(y_labels)]
        pred_centroids = [region.centroid for region in measure.regionprops(pred_labels)]

        # 计算真阳性、假阳性、漏检区域

        tmp_tp = 0
        # print(pred_labels)
        for pl in np.unique(pred_labels):
            pred_region = pred_labels == pl
            if np.sum(pred_region & (y_labels > 0)) > 0:
                total_tp += 1
                tmp_tp += 1
            else:
                total_fp += 1

        for fl in np.unique(y_labels):
            y_region = y_labels == fl
            if np.sum(y_region & (pred_labels > 0)) == 0:
                total_fn += 1
        
   
        # total_fn += len(np.unique(y_labels)) - tmp_tp
        # print(f"total_tp:{total_tp},total_fp:{total_fp},total_fn:{total_fn}")
    
        if id==0:
        #     print(f'pred_labels:{np.unique(pred_labels)}')
        #     print(f'y_labels:{np.unique(y_labels)}'
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(x.cpu().numpy()[0].squeeze(), cmap='gray')
            ax[0].set_title('Input')

            ax[1].imshow(y_seg.squeeze(0).squeeze(0),cmap="gray", alpha=0.5)
            ax[1].imshow(pred_seg.squeeze(0).squeeze(0), cmap='Purples', alpha=0.2)
            ax[1].set_title('Predictions vs Ground Truth')

            ax[1].scatter([p[1] for p in y_centroids], [p[0] for p in y_centroids], c='r', s=1)
            ax[1].scatter([p[1] for p in pred_centroids], [p[0] for p in pred_centroids], c='b', s=1)
            # ax[2].imshow(pred_labels)
            # ax[3].imshow(y_labels)
            print(f"len_y:{len(y_centroids)},len_pred:{len(pred_centroids)},len:{len(np.unique(y_labels))},total_tp:{total_tp},total_fp:{total_fp},total_fn:{total_fn}")

            plt.show()
            # plt.savefig(f'img/{z}.png')
            plt.close()
            # print(total_fp,total_tp,total_fn)
        if id%100==0:
            print(id)
        id+=1

recall = total_tp / (total_tp + total_fn)  
precision = total_tp / (total_tp + total_fp)

print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f"total_tp:{total_tp},total_fp:{total_fp},total_fn:{total_fn},total_tn:{total_tn}")

