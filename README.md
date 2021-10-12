# [Person ReID](https://github.com/layumi/Person_reID_baseline_pytorch) model compression ([PyTorch](https://pytorch.org/))

Note: Original code is heavily borrowed from https://github.com/layumi/Person_reID_baseline_pytorch

## LPCV Person ReID model compression Pipeline

We did the following steps in our work:
1. Train Person ReID model (ResNet18) from pretrained model. Please check [train.py](./Person_reID_baseline_pytorch/train.py)
2. Prune ReID model with [NNI](https://github.com/microsoft/nni). Please check [train.py](./Person_reID_baseline_pytorch/train.py)
3. Delete the pruned channels and export the pruned model. Please check [export_pruned_resnet.py](./Person_reID_baseline_pytorch/prune/export_pruned_resnet.py)
4. Finetune the model and iterate step 2 and step 3  
5. Finetune the model with new resized image
6. Quantize the ReID model with pytorch quantization package. Please check [quantize](.Person_reID_baseline_pytorch/quantize)

## Training Person ReID Model
### Dataset

Multiple Datasets

In LPCV video track challenge, we merged datasets with multiple benchmarks. The benchmarks we used were [Market-1501](https://paperswithcode.com/dataset/market-1501), [DukeMTMC-reID](https://paperswithcode.com/dataset/dukemtmc-reid) and [CUHK Person Re-identification Datasets](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). The merged dataset 
contains a total of 2424 classes of people. 

### Train model
We train our Person ReID model following the pipeline of [Person ReID](https://github.com/layumi/Person_reID_baseline_pytorch) with a slight modification of using ResNet18 instead of using ResNet50.
1. Train from pretrained model using circle loss with specified learning rate, warm up epoch and batchsize
    ```sh
    python train.py --circle --lr=0.01 --batchsize 32 --warm_epoch=5
    ```

2. More parameter explanation:
   
    ```--gpu_ids``` which gpu to run<br>
    ```--circle``` using circle loss<br>
    ```--name``` the directory name of trained model<br>
    ```--data_dir``` the path of the testing data<br>
    ```--train_all``` Using all images to train<br>
    ```--erasing_p``` using random erasing probability<br>

   
### Delete pruned channels and export the model
After training the ReID model, we prune the model using NNI (Neural Network Intelligence) toolkit. Since NNI only sets the weight of the pruned channels to zero, we need to delete the pruned channels manually.

1. Prune and export the model.
    ```sh
    python export_pruned_resnet.py
    ```

2. You can finetune the pruned model by running train.py on the pruned model.
   
### Quantize model
After pruning model, we quantize the model using Pytorch QNNPACK. Both static quantization and Quantization Aware Training are provided.
1. Static Quantize model
    ```sh
    python quantize.py
    ```
    After running quantize.py, a quantized ReID model is saved in torchscript format.
    
2. Quantization Aware Training:  
  
    ```sh
    python qat.py --batchsize=32
    ```
    Quantization Aware Training (QAT) involves training during quantization process. In our experiment, QAT usually yields a slightly better accuracy compared to 
    static quantization.

