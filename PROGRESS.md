### Profiling(05/14)

|    Section    |    Comments    | Speed(FPS) |
| :-----------: | :------------: | :--------: |
| Preprocessing |  Crop&resize   |    ~500    |
|   Detection   |    Yolo&NMS    |     ~6     |
|   Tracking    |   Deep sort    |    ~14     |
|     ReID      |  Deep feature  |    ~15     |
|    Tracker    | KF&Association |    ~200    |

### Experiments


| Author@Date |          Comments          | Performance(FPS) |
| :---------: | :------------------------: | :--------------: |
|  Tom@07/02  |   Yolov5l, cpu, baseline   |       ~0.7       |



### Quantization

| Author@Date |             Comments              |    Speed(FPS)    |    Size(KB)    |    mAP    |
| :---------: | :-------------------------------: | :--------------: | :--------------: | :--------------: |
| Baseline   | Resnet18                              |       ??         |     47901.5     |     0.650     |
| Leo@08/06  | best_mode1_qnn.torchscript, 0.5 (RPI) |       21.7       |      2891.1     |      0.561     |  
| Leo@08/06  | FPGM_0.875_qnn.torchscript, 0.125 (RPI) |       66.6       |      311.0      |      0.383    |   

### Correctness

|        Name              | Dataset       | Rank@1  | mAP     | 
| :----------------:       | :----:        | :-----: | :-----: | 
|      Baseline (ResNet18) |  Market-1501  |  0.851  |  0.637  |  
|      Baseline            |  SampleVideo       |  0.964  |  0.816  |
|      Baseline            |  Occlusion       |  0.78  |  0.642  |
|      Baseline            |  Hard      |  0.82  |  0.744  |
|      Compressed (old)    |  Market-1501  |  0.622  |  0.373  |
|      Compressed          |  Market-1501  |  0.712  |  0.442  |
|      Compressed          |  SampleVideo       |  0.952  |  0.803  |
|      Compressed          |  Occlusion       |  0.804  |  0.692  |
|      Compressed          |  Hard      |  0.833  |  0.727  |

