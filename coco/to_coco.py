import json
import os
import pandas as pd

root = "D:\workspace\AI4SCI\smart\code\Pytorch-UNet\data/"
# , "2,1000,500/", "cell/", "epl/", "gfp/","0.05/","0.05,1600/"
folders = ["2,1000/", "2,1000,500/", "cell/", "epl/", "gfp/","0.05/","0.05,1600/"]
folders_path = [os.path.join(root, folder) for folder in folders]

# 创建COCO数据结构
coco_data = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

# 定义类别信息
category_info = {
    "id": 1,
    "name": "point",
    "supercategory": "none"
}
coco_data["categories"].append(category_info)
annotation_id=0
# 遍历文件夹
image_id = 0
for folder_path in folders_path:
    csv_path = os.path.join(folder_path, "frame_logger.csv")

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 根据图像分组
    grouped = df.groupby("frame")

    # 添加图像信息和目标信息
    ids=[]
    for frame, group in grouped:
        ids.append(frame-1)
        # 添加图像信息
        image_data = {
            "id": image_id,
            "width": 128,  # 图像宽度
            "height": 128,  # 图像高度
            "file_name": f"{folder_path}/images/image_{frame-1}.jpg"  # 可根据实际情况修改文件名
        }
        coco_data["images"].append(image_data)

        # 添加目标信息
        for _, row in group.iterrows():
            # annotation_id= row["id"]
            annotation_data = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # 类别ID为1，表示"point"
                "bbox": [row["x [px]"], row["y [px]"], 5, 5],
                "area": 5** 2,
                "iscrowd": 0,
                "attributes": {},
                "metadata": {},
            }
            annotation_id+=1
            coco_data["annotations"].append(annotation_data)
        

        # 增加image_id计数
        image_id += 1
    for i in range(1000):
        if i not in ids:
            image_data = {
                "id": image_id,
                "width": 128,  # 图像宽度
                "height": 128,  # 图像高度
                "file_name": f"{folder_path}/images/image_{i}.jpg"  # 可根据实际情况修改文件名
            }
            coco_data["images"].append(image_data)
            image_id += 1

# 保存为JSON文件
with open("coco/coco_dataset.json", "w") as f:
    json.dump(coco_data, f)