# 21LPCV-UAV-Solution

## data synthesize using coco dataset
This is an example of generating ball occluded example images. The output folder will be synthesized images and labels (ball/person). The label format will be each row is class x_center y_center width height format.
Box coordinates are normalized xywh format (from 0 - 1). 
To use it:
1. Download coco dataset 2017 and install cocoapi via https://github.com/cocodataset/cocoapi 
2. Download ball examples from https://drive.google.com/file/d/11CdkcZ-ubDcSGm2AvaIa8pYo2u-HWoJ5/view?usp=sharing
3. Modify directories in person_imgs_selector.py to match your coco dataset downloaded directories at 

```python 
P = "/media/pi/ssd1/coco"
IMAGE_P = os.path.join(P, "images")
dataDir='/media/pi/ssd1/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
```
4. Modify 



