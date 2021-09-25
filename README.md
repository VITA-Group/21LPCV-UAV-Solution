# 21LPCV-UAV-Solution

## data synthesize using coco dataset
This is an example of generating ball person images. The output folder will be synthesized images and labels (ball/person). The label format will be each row is class x_center y_center width height format.
Box coordinates are normalized xywh format (from 0 - 1). 

To use it, you only need **person_imgs_selector.py** and **ball_synth_v2.py** files. One is to select person examples, the other is used to paste ball into person instances. You just need to slightly modify directories according to below instructions. 

1. Download coco dataset 2017 and install cocoapi via https://github.com/cocodataset/cocoapi 
2. Download ball examples from https://drive.google.com/file/d/11CdkcZ-ubDcSGm2AvaIa8pYo2u-HWoJ5/view?usp=sharing
3. Modify directories in person_imgs_selector.py to match your coco dataset downloaded directories at https://github.com/wuzhenyusjtu/21LPCV-UAV-Solution/blob/7c5804b21a647b969ff6385a8c070419ef475fda/person_imgs_selector.py#L11

```python 
P = "/media/pi/ssd1/coco"
IMAGE_P = os.path.join(P, "images")
dataDir='/media/pi/ssd1/coco'
dataType='train2017'
```
4. Modify filter condition as your wish in person_imgs_selector.py
5. check output images folder if you satisfy with selected samples
6. run person_imgs_selector.py to generate coresponding image and label folders for ball_synth_v2.py to use
7. Modify ball_synth_v2.py's coco directory at
```python 
# setup coco directory
dataDir = '/media/pi/ssd1/coco'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
P = "/media/pi/ssd1/coco"
```
modify ball directories to ball samples downloaded at https://github.com/wuzhenyusjtu/21LPCV-UAV-Solution/blob/7c5804b21a647b969ff6385a8c070419ef475fda/ball_synth_v2.py#L120
```python
mypath_balls = '/media/pi/ssd1/cocostuffapi/PythonAPI/balls/balls'
```
8. run ball_synth_v2.py to generate occluded ball person samples. 






