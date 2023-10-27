from PIL import Image, ImageEnhance
import os
import cv2
import imageio
import numpy as np
root = "./data/"
# "2,1000/", "2,1000,500/", , "epl/", "gfp/"
folders = ["2,1000/", "2,1000,500/","cell/", "epl/", "gfp/"]
folders_path = [os.path.join(root, folder) for folder in folders]

for folder_path in folders_path:
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    os.makedirs(os.path.join(folder_path, "images"), exist_ok=True)
    for file in files:
        if file.endswith(".tif"):
            tif_path = os.path.join(folder_path, file)
            tif_image = imageio.get_reader(tif_path)
            print(tif_image)
            a=tif_image.get_next_data()
            print(a.dtype)
            print(a.ndim)
            print(a.max(),a.min())
            # print(a)
            data=a
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

      
            data /= max_uint
            print(data.max(),data.min())
            
            data=(data-data.min())/(data.max()-data.min())
            for frame in range(data.shape[0]):
                tif_frame=data[frame]
                # 获取最小和最大像素值

                min_pixel = tif_frame.min()
                max_pixel = tif_frame.max()
                # print(f"File: {file}, Frame: {frame}, Min Pixel Value: {min_pixel}, Max Pixel Value: {max_pixel}")

                # 转换为灰度图像
                grayscale_frame = (tif_frame * 2 ** 16 - 1).astype(np.uint16)
                min_pixel = grayscale_frame.min()
                max_pixel = grayscale_frame.max()
                # print(f"File: {file}, Frame: {frame}, Min Pixel Value: {min_pixel}, Max Pixel Value: {max_pixel}")

                # grayscale_frame = tif_frame  # 无需转换
                
                # 生成PNG或JPG文件名
                image_name, _ = os.path.splitext(file)
                image_name = f"image_{frame}.png"  # 或者使用".jpg"替代".png"
                image_save_path = os.path.join(folder_path, "images", image_name)
                # 将当前帧保存为PNG或JPG
                imageio.imsave(image_save_path, grayscale_frame)