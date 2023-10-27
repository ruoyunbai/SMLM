import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
origin=128
max=128
# max=256
scale=max/origin
x_max = max
y_max = max
# data_path='./data/2,1000,500/'
data_path='./data/gfp/'
# 220 chpt
def generate_binary_map(target_loc, x_max=64, y_max=64):
    """Generates a binary map with a single 1 at the target location.

    Parameters
    ----
    target_loc : tuple of float
        A tuple containing the (x, y) coordinates of the target point.
    x_max : int
        Number of horizontal pixels of the density map (default 64)
    y_max : int
        Number of vertical pixels of the density map (default 64)

    Returns
    ----
    binary_map : numpy array of int
        x_max-by-y_max array containing the binary map with a single 1 at
        the rounded target location.
    """
    # print(f"min_x:{target_loc[:,0].min()},max_x:{target_loc[:,0].max()},min_y:{target_loc[:,1].min()},max_y:{target_loc[:,1].max()}")
    binary_map = np.zeros((x_max, y_max))
    for i in range(target_loc.shape[0]):
        x, y = target_loc[i]
        rounded_x = int(round(x*scale))
        rounded_y = int(round(y*scale))
        if rounded_x==x_max:
            rounded_x=x_max-1
        if rounded_y==y_max:
            rounded_y=y_max-1
        if 0 <= rounded_x < x_max and 0 <= rounded_y < y_max:
            binary_map[rounded_y, rounded_x] = 1
        else:
            print(f"i:{i},x:{x},y:{y},rounded_x:{rounded_x},rounded_y:{rounded_y}")
            print("Target location is outside the binary map.")

    return binary_map




# target_location = (64.5, 64.5)  # Set the target location (non-integer) here

# binary_map = generate_binary_map(target_location, x_max, y_max)
# image_name = 'binary_map.jpg'
# plt.imsave(image_name, binary_map, cmap='gray')


# df=pd.read_csv("./data/1/frame_logger.csv")
df=pd.read_csv(f"{data_path}frame_logger.csv")
# df=pd.read_csv("./data/2,1000,500/frame_logger.csv")
# 分组
grouped = df.groupby('frame') 
stack=[]
diff=0
import os
# if not os.path.exists('./data/2,1000,500/ground_truth_bin'):
#   os.makedirs('./data/2,1000,500/ground_truth_bin')
if not os.path.exists(f'{data_path}ground_truth_bin'):
  os.makedirs(f'{data_path}ground_truth_bin')
for frame, group in grouped:
  
#   locs = group[['x','y']].values
  locs = group[['x [px]','y [px]']].values

  
  genMap = generate_binary_map(locs, x_max, y_max) 
  stack.append(genMap)
  print(f"{frame-1 } {len(locs)} {genMap.sum()} {genMap.shape}")
  # 保存图像
  diff+=genMap.sum()-len(locs)
#   image_name = './data/2,1000,500/ground_truth_bin/frame_0.1_{}.jpg'.format(frame-1) 
  image_name = f'{data_path}ground_truth_bin/frame{frame-1}.jpg'
  plt.imsave(image_name, genMap,cmap='gray')
#   plt.imshow(genMap,cmap='gray')
#   plt.scatter(locs[:,0],locs[:,1])
#   plt.show()
print(f"diff:{diff}")
