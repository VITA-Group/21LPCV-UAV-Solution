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
|  Tom@07/02  | **Yolov5s**, cpu, baseline |       ~1.5       |
|  Tom@05/11  |   Crop image,CPU/2080Ti    |     4.84/43      |
|  Tom@05/26  |       Yolov5l,2080Ti       |        58        |
|  Tom@05/26  |       Yolov5s,2080Ti       |      64.05       |


### Quantization

| Author@Date |             Comments              |    Speed(FPS)    |
| :---------: | :-------------------------------: | :--------------: |
| Xin@08/06(416)   | yolov5s_qnnpack.torchscript(pi)       |       8.5        |
| Xin@08/06(640)   | yolov5s_qnnpack.torchscript(pi)       |       3.9        |
| John@07/02  |    yolov5s_fbgemm.torchscript     |       6.58       |
| John@07/02  | yolov5s_fbgemm.torchscript, 0.125 |       7.64       |

### Correctness

|        Name        | 4p(1m) | 5p2(2m) | 5p4(1m) | 5p5(1m) |  7p(2m)   | Avg  |
| :----------------: | :----: | :-----: | :-----: | :-----: | :-------: | :--: |
|      Baseline      |  1.0   |  0.97   |  0.95   |  0.46   |   0.15    | 0.66 |
|        Crop        |  1.0   |  0.97   |  0.83   |  0.47   |   0.21    | 0.66 |
|    YOLOv5l,coco    |  1.0   |  0.84   |  0.86   |  0.47   |   0.24    | 0.64 |
|    YOLOv5l,vip     |  1.0   |   1.0   |  0.82   |  0.56   | 0.18(val) | 0.67 |
| YOLOv5l, vip+coco  |  1.0   |   1.0   |  0.88   |  0.52   |   0.23    | 0.69 |
| YOLOv5l, sync+coco |  1.0   |   1.0   |  0.88   |  0.49   |   0.23    | ???  |
