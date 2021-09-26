# 21LPCV-UAV-Solution

## Data synthesize using coco dataset


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

## Detection architecture and compression  


Note: Original code borrow from https://github.com/ultralytics/yolov5

### LPCV YOLOv5 Detection Part Pipeline

We did the following steps in our work:
1. Train YOLOv5 detection model from pretrained model with ball-person Dataset. Please check train.py
2. Prune YOLOv5 model with [NNI](https://github.com/microsoft/nni). Please check [prune](./prune)
3. Finetune pruned model with Merged Dataset. Please check train.py  
4. Quantize YOLOv5 model with pytorch quantization package. Please check [quantize](./quantize)

### LPCV YOLOv5 Standard Training  
#### Train model
We train YOLOv5 following the official yolov5 pipeline except with our own VIP dataset.
1. Train from pretrained model with VIP dataset (Multi-GPU DataDistributedParallel)
    ```sh
    python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 256 --weights weights/yolov5s.pt\
    --device 0,1,2,3,4,5,6,7 --cfg models/yolos_new.yaml
    ```
2. Train from scratch with VIP dataset (Multi-GPU DataDistributedParallel)
    ```sh
    python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 256 --weights ' '\
    --device 0,1,2,3,4,5,6,7 --cfg models/yolos_new.yaml
    ```
3. Parameter explanation:
   
    ```--data``` path to the vip config file<br>
    ```--cfg``` path to the yolov5 config file<br>
    ```--batch``` set batch size for training<br>
    ```--nproc_per_node``` set number of processes per node<br>
    ```--device``` set device id use<br>
    ```--weights``` path to the pretrained model, '' if train from scratch<br>
    ```--epochs``` how many epochs to train<br>
   
#### Train pruned model
After training model with vip dataset, we prune the model using NNI (Neural Network Intelligence) toolkit. 
1. Train pruned model with VIP dataset.
    ```sh
    python3 -m torch.distributed.launch prune_yolov5.py --data data/vip_new.yaml --weights weights/trained_yolov5s.pt \
    --batch-size 256 --epochs 60 --prune L1FilterPruner --base-algo l1 --device 0,1,2,3,4,5,6,7
    ```

2. Parameter explanation:
   
    ```--data``` path to the vip config file<br>
    ```--cfg``` path to the yolov5 config file<br>
    ```--batch-size``` set batch size for training<br>
    ```--nproc_per_node``` set number of processes per node<br>
    ```--device``` set device id use<br>
    ```--weights``` path to the pretrained model, '' if train from scratch<br>
    ```--epochs``` how many epochs to train<br>
    ```--prune``` set pruner, L1 Pruner for example<br>
    ```--base-algo``` select base algorithm, l1 for example<br>
   
#### Quantized model
After pruning model, we quantize the model using Pytorch QNNPACK.
1. Static Quantize model
    ```sh
    python3 static_quantize_yolov5.py --calibrate-folder '' --save-dir weights/ \
    --pretrain-model weights/trained_yolov5s.pt --backend qnnpack
    ```

2. Quantization Aware Training:  
  
    ```sh
    python3 qat_yolov5.py --calibrate-folder '' --save-dir weights/ --backends qnnpack --epochs 60\
    --device 0 --batch-size 64 --weights weights/trained_yolov5s.pt --data ''
    ```  

3. Parameter explanation:
   
    ```--calibrate-folder``` path to the calibration folder<br>
    ```--save-dir``` path to the directory to be saved<br>
    ```--pretrain-model``` path to the pretrained model to quantize<br>
    ```--backend``` set the backend for quantization<br>
    ```--weights``` path to the pretrained model to quantize<br>
    ```--device``` set device id use<br>
    ```--batch-size``` set batch size for quantization aware training<br>
    ```--data``` path to the vip config file<br>
    









