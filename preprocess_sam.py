import pandas as pd
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import shutil
def generate_map(locs, x_max=64, y_max=64, sigma=1):
    """Generates a density map from an array of fluorophore positions

    Parameters
    ----
    locs : numpy array of double
        N-by-2 ndarray containing the horizontal and vertical positions of the
        fluorophores in the frame, in pixels (but with subpixel precision)
    x_max : int
        Number of horizontal pixels of the density map (default 64)
    y_max : int
        Number of vertical pixels of the density map (default 64)

    Returns
    ----
    genMap : numpy array of double
        x_max-by-y_max array containing the values for each pixel of the
        density map
    """

    # meshgrid
    x, y = np.meshgrid(range(x_max), range(y_max))
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    grid = np.hstack((x,y))

    # create a map
    genMap = np.zeros((grid.shape[0],1))

    for k in range(locs.shape[0]):
            # Sum the mvn at each position for each pixel
            mvn = multivariate_normal.pdf(grid + 0.5, locs[k,:], cov=sigma)
            genMap += mvn.reshape(-1,1)
    genMap = genMap.reshape((x_max, y_max))
    threshold=genMap.mean()+1*genMap.std()
    genMap=genMap>threshold
    return genMap

x_max=128
y_max=128
data_paths=['./data/0.05/','./data/0.05,1600/','./data/2,1000/','./data/2,1000,500/','./data/cell/','./data/epl/','./data/gfp/']
# data_path = './data/cell/'
for data_path in data_paths:
  print(data_path)
  # data_path = './data/1/frame_logger.csv'
  # df=pd.read_csv("./data/1/frame_logger.csv")
  df=pd.read_csv(f"{data_path}frame_logger.csv")
  # df=pd.read_csv("./data/2,1000,500/frame_logger.csv")
  # 分组
  grouped = df.groupby('frame') 
  stack=[]
  import os
  # if not os.path.exists('./data/2,1000,500/ground_truth'):
  #   os.makedirs('./data/2,1000,500/ground_truth')
  # if not os.path.exists(f"{data_path}ground_truth"):
  shutil.rmtree(f"{data_path}/ground_truth_sam",ignore_errors=True)
  os.makedirs(f"{data_path}/ground_truth_sam",exist_ok=True)
  frame_ids=[]
  for frame, group in grouped:
    frame_ids.append(frame-1)
  #   locs = group[['x','y']].values
    locs = group[['x [px]','y [px]']].values

    
    genMap = generate_map(locs, x_max, y_max) 
    genMap=genMap.astype(np.uint8)
    # print(genMap.dtype)
    stack.append(genMap)
    # print(f"{frame-1 } {len(locs)} {genMap.sum()} {genMap.shape}")
    # 保存图像
  #   image_name = './data/2,1000,500/ground_truth/frame_0.1_{}.jpg'.format(frame-1) 
    image_name = f'{data_path}ground_truth_sam/frame_{frame-1}'
    # plt.imsave(image_name, genMap,cmap='gray')
    np.save(image_name,genMap.astype(np.uint8))
    if frame%100==0:
      print(frame)
    # break

  # tifffile.imsave('density_stack.tif', stack)
  for i in range(1000):
      if not i in frame_ids:
          image_name = f'{data_path}ground_truth_sam/frame_{i}'
          print(image_name)
          np.save(image_name, np.zeros((x_max,y_max)).astype(np.uint8))
          # plt.imsave(image_name, np.zeros((x_max,y_max)),cmap='gray')
          

