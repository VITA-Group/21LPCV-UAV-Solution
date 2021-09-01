# 2021LPCVC-UAVVideoTrack@VITA (https://vita-group.github.io/)
## 1. Challenge Introduction
+ Description: https://lpcv.ai/2021LPCVC/video-track
+ Sample Videos: https://drive.google.com/drive/folders/1S6kfqSG8AJpoj-y-4-nIagfmL7FpVTOf
+ Sample Solution: https://github.com/lpcvai/21LPCVC-UAV_VIdeo_Track-Sample-Solution
+ Referee System for Evaluation: https://github.com/lpcvai/2021-LPCVC-Referee
+ Annotated Sample Videos: https://drive.google.com/drive/folders/1Gj8iO99fPp3j7q_gvuLEb_EKbeZlamq_
## 2. Efficient Neural Network
### 2.1. Literature Review
#### 2.1.1. Compression & Acceleration
+ [Arxiv, 2018] Recent Advances in Efficient Computation of Deep Convolutional Neural Networks, https://arxiv.org/abs/1802.00939
+ [IEEE Signal Processing Magzine, 2020] A Survey of Model Compression and Acceleration for Deep Neural Networks, https://arxiv.org/abs/1710.09282
+ [Proceedings of the IEEE, 2020] Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey, https://ieeexplore.ieee.org/document/9043731
+ [Arxiv, 2021] Pruning and Quantization for Deep Neural Network Acceleration: A Survey, https://arxiv.org/abs/2101.09671
#### 2.1.2. Quantization
+ [Arxiv, 2018] Quantizing deep convolutional networks for efficient inference: A whitepaper, https://arxiv.org/abs/1806.08342
+ [Arxiv, 2018] A Survey on Methods and Theories of Quantized Neural Networks, https://arxiv.org/abs/1808.04752
+ [CVPR, 2018] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, https://arxiv.org/abs/1712.05877
+ [NIPS, 2018] Scalable Methods for 8-bit Training of Neural Networks, https://arxiv.org/abs/1805.11046
+ [ICCV, 2019] Data-Free Quantization Through Weight Equalization and Bias Correction, https://arxiv.org/abs/1906.04721
+ [Arxiv, 2021] A Survey of Quantization Methods for Efficient Neural Network Inference, https://arxiv.org/abs/2103.13630
#### 2.1.3. Pruning
+ [Arxiv, 2020] Pruning Algorithms to Accelerate Convolutional Neural Networks for Edge Applications: A Survey, https://arxiv.org/pdf/2005.04275.pdf
+ [ICML, 2021] Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework, https://arxiv.org/abs/2010.04879
+ [NIPS, 2020] Movement Pruning: Adaptive Sparsity by Fine-Tuning, https://arxiv.org/abs/2005.07683
#### 2.1.4. Adpative Inference
+ [Arxiv, 2021] Dynamic Neural Networks: A Survey, https://arxiv.org/abs/2102.04906
### 2.2. Open-source PyTorch Package for Model Compression & Hardware Deployment
+ [Intel AI Lab] Neural Network Distiller, https://github.com/IntelLabs/distiller
+ [Microsoft Research] Neural Network Intelligence, https://github.com/microsoft/nni
+ [FaceBook Research] D2Go, https://github.com/facebookresearch/d2go
### 2.3. Github Repos
+ https://github.com/he-y/Awesome-Pruning
+ https://github.com/htqin/awesome-model-quantization
+ https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression

## 3. Detection & Tracking
### 3.1. Literature Review
#### 3.1.1. Detection & Tracking Survey
+ [Arxiv, 2019] Deep Learning in Video Multi-Object Tracking: A Survey, https://arxiv.org/abs/1907.12740
+ [IEEE TITS] Deep Learning for Visual Tracking: A Comprehensive Survey, https://arxiv.org/abs/1912.00535
+ [Neurocomputing, 2018] Computer Vision and Deep Learning Techniques for Pedestrian Detection and Tracking: A Survey
#### 3.1.2. One-Stage Detector
+ [CVPR, 2016] You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
+ [CVPR, 2017] YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
+ [Arxiv, 2018] YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767
+ [Arxiv, 2020] YOLOv4: Optimal Speed and Accuracy of Object Detection, https://arxiv.org/abs/2004.10934
+ [CVPR, 2021] You Only Look One-level Feature, https://arxiv.org/abs/2103.09460
+ [Arxiv, 2021] You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection, https://arxiv.org/abs/2106.00666
+ [Arxiv, 2017] Fast YOLO: A Fast You Only Look Once System for Real-time Embedded Object Detection in Video, https://arxiv.org/abs/1709.05943
#### 3.1.3. Online Tracking
+ [ICIP, 2016] Simple Online and Realtime Tracking, https://arxiv.org/abs/1602.00763
+ [ICIP, 2017] Simple Online and Realtime Tracking with a Deep Association Metric, https://arxiv.org/abs/1703.07402
#### 3.1.4. Person ReID
+ [TPAMI, 2021] Deep Learning for Person Re-identification: A Survey and Outlook, https://arxiv.org/abs/2001.04193
### 3.2. Github Repos
+ https://github.com/amusi/awesome-object-detection
+ https://github.com/kuanhungchen/awesome-tiny-object-detection
+ https://github.com/luanshiyinyang/awesome-multiple-object-tracking
+ https://github.com/sdsy888/Awesome-Object-Tracking
+ https://github.com/bismex/Awesome-person-re-identification

## 4. Image Composition & Harmonization
### 4.1. Literature Review
#### 4.1.1. Image Composition
+ [Arxiv, 2021] Crop-Transform-Paste: Self-Supervised Learning for Visual Tracking, https://arxiv.org/abs/2106.10900
+ [Arxiv, 2021] Survey: Image Mixing and Deleting for Data Augmentation, https://arxiv.org/abs/2106.07085
+ [CVPR, 2021] Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation, https://arxiv.org/abs/2012.07177
#### 4.1.2. Image Harmanization
+ [Arxiv, 2021] Making Images Real Again: A Comprehensive Survey on Deep Image Composition, https://arxiv.org/abs/2106.14490
#### 4.1.3. Debiased Training
+ [ICLR, 2019] ImageNet-trained CNNs Are Biased Towards Texture; Increasing Shape Bias Improves Accuracy and Robustness, https://arxiv.org/abs/1811.12231
+ [ICLR, 2021] Shape-Texture Debiased Neural Network Training, https://arxiv.org/abs/2010.05981
+ [Arxiv, 2021] Small In-Distribution Changes in 3D Perspective and Lighting Fool Both CNNs and Transformers, https://arxiv.org/pdf/2106.16198.pdf
### 4.2. Github Repos
+ https://github.com/bcmi/Awesome-Image-Harmonization
+ https://github.com/CrazyVertigo/awesome-data-augmentation
+ https://github.com/bcmi/Awesome-Image-Composition
