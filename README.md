# 21LPCV-UAV-Solution

## data synthesize using coco dataset


![000000147772](https://user-images.githubusercontent.com/35612650/134788860-c8693c55-45d6-4ccd-bd35-01209994258d.jpg)


![000000147772](https://user-images.githubusercontent.com/35612650/134788842-60c95c9d-95a8-4170-babb-839ea489052f.jpg)

This is an example of generating ball person images. The output folder will be synthesized images and labels (ball/person). The label format will be each row is class x_center y_center width height format.
Box coordinates are normalized xywh format (from 0 - 1). 

To use it, you only need **person_imgs_selector.py** and **ball_synth_v2.py** files. One is to select person examples, the other is used to paste ball into person instances. You just need to slightly modify directories according to below instructions. 

1. Download coco dataset 2017 and install cocoapi via https://github.com/cocodataset/cocoapi 
2. Download ball examples from https://drive.google.com/file/d/11CdkcZ-ubDcSGm2AvaIa8pYo2u-HWoJ5/view?usp=sharing
3. Modify directories in person_imgs_selector.py to match your coco dataset downloaded directories at 

```python 
P = "/media/pi/ssd1/coco"
IMAGE_P = os.path.join(P, "images")
dataDir='/media/pi/ssd1/coco'
dataType='train2017'
```
4. Modify filter condition as your wish in person_imgs_selector.py
5. run person_imgs_selector.py to generate coresponding image and label folders for ball_synth_v2.py to use
6. check output images folder if you satisfy with selected samples
7. Modify ball_synth_v2.py's coco directory at
```python 
# setup coco directory
dataDir = '/media/pi/ssd1/coco'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
P = "/media/pi/ssd1/coco"
```
modify ball directories to ball samples downloaded before at 
```python
mypath_balls = '/media/pi/ssd1/cocostuffapi/PythonAPI/balls/balls'
```
8. run ball_synth_v2.py to generate occluded ball person samples. 






