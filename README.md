# Yolov5 实现武器检测
### 疫情期间在中科信息实习期间的部分成果。
### 使用模型时yolov5：https://github.com/ultralytics/yolov5
### 使用数据来自kaggle：https://www.kaggle.com/jubaerad/weapons-in-images-segmented-videos

## 数据集如图所示
![Image text](https://github.com/SoulkeyZhang/weapon-detect-/blob/master/runs/exp15/test_batch0_gt.jpg)

## 训练结果如图
![Image text](https://github.com/SoulkeyZhang/weapon-detect-/blob/master/runs/exp15/results.png)

## 检测结果
### 视频检测的结果
![Image text](https://github.com/SoulkeyZhang/weapon-detect-/blob/master/UI_images/test_cam.png)
![Image text](https://github.com/SoulkeyZhang/weapon-detect-/blob/master/UI_images/test_cam2.png)

## 利用PyQt5实现的UI
### UI界面如下图所示，截至2020/08/17 该UI还存在bug
![Image text](https://github.com/SoulkeyZhang/weapon-detect-/blob/master/UI_images/ui.png)
### 具体实现在weaponUI.py中