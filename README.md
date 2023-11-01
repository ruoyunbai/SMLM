### 日志
2023.10.26：
>	加入SwinUnet、MedSam（可以测定不同数据finetue的结果）、NNUnet模型对比
	from scratch和finetune效果对比
	数据使用模拟数据8：1：1
	在test数据集测定评测指标
	形成大表，竖行是模型+scratch/finetune，横行有召回率（荧光点被找到的比率，粗略一点，只要范围有重合就算找到）、精度（预测的点中正确点的比率）

>	形成大表之后选用效果好的，再在真实数据上测试。应用domain adaptation、domain generalization加强泛化



### 文件组织
有unet的代码和trans_unet的代码，基本一样
- ```/tran2.py``` 训练unet
- ```/smart_predict_real.py``` unet在真实数据上的结果
- ```/smart_predict_real2.py``` 两个unet在真实数据上的结果（废弃）
- ```/smart_predict_validate.py``` unet在验证集上的结果
- ```/smart_predict_validate.py``` 两个Unet在验证集的结果（废弃）
- ```/trans_tran.py``` 训练trans_unet
- ```/trans_smart_predict_validate.py``` trans_unet训练验证集上测试
-  ```/trans_smart_predict_predict_real.py``` trans_unet真实数据集测试
- ```/data``` 训练数据
- ```/chkps```训练记录和checkpoints 
- ```/unet``` unet模型
-  ```/trans_unet``` trans_unet模型
  
##### data下载：
>https://bhpan.buaa.edu.cn/link/AA6E92C9F8BBCF462E9ADBDC61CCC4F735
文件夹名：smart
有效期限：2024-07-12 13:50 
### 任务
```smart_predict_validate.py```增加召回率、精度指标

gwx:SwinUnet、MedSam

nr:NNUnet




	python train swin_train.py &	python train train2.py &	python train trans_train.py
