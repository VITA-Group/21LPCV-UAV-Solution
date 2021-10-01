from __future__ import print_function, division
import sys
sys.path.insert(0, '..')
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import copy
#from models.pruned_model import ft_net
from models.model import ft_net
from random_erasing import RandomErasing
import scipy.io
import math
from shutil import copyfile

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ResNet18_Pruned', type=str, help='output model name')
parser.add_argument('--data_dir',default='../../Market-1501/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_false', help='use all training data' )
parser.add_argument('--color_jitter', action='store_false', help='use color jitter in training' )
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_false', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_false', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_false', help='use PCB+ResNet50' )
parser.add_argument('--circle', action='store_false', help='use Circle loss' )
parser.add_argument('--fp16', action='store_false', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--multi', action='store_false', help='use multiple query' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
opt = parser.parse_args()

fp16 = False
opt.fp16 = False
opt.circle = False
opt.PCB = False
opt.use_NAS = False
opt.use_dense = False
opt.color_jitter = False
opt.tran_all = False
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
'''if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True'''
######################################################################
# Load Data
# ---------
#

transform_train_list = [
    #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256,128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = False

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)

####################################################################################
class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):

        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32.model.conv1(x)
        x = self.model_fp32.model.bn1(x)
        x = self.model_fp32.model.relu(x)
        x = self.model_fp32.model.maxpool(x)

        x = self.model_fp32.model.layer1(x)
        x = self.model_fp32.model.layer2(x)
        x = self.model_fp32.model.layer3(x)
        x = self.model_fp32.model.layer4(x)

        x = self.model_fp32.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.model_fp32.classifier(x)
        x = self.dequant(x)
        return x


####################################################################################
def model_equivalence(model_1,
                      model_2,
                      device,
                      rtol=1e-05,
                      atol=1e-08,
                      num_tests=100,
                      input_size=(1, 3, 32, 32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True
####################################################################################
opt.nclasses = len(class_names)

def load_network(network):
    name = 'ResNet18_Pruned'
    file_name = 'pruned_FPGM_0.875.pth'
    # save_path = os.path.join('./model',name,'net_%s.pth'%file_num)
    save_path = os.path.join('../model',name,file_name)
    print('Loading model: %s'%file_name)
    network.load_state_dict(torch.load(save_path))
    return network


def save_network(network, epoch_label):
    name = 'ResNet18_Qt'
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

def save_torchscript_model(model, model_dir, model_filename):
    device = torch.device("cpu")
    dummy_input = torch.randn([256, 3, 256, 128]).to(device)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.trace(model, dummy_input, strict=True), model_filepath)         # torch.jit.trace


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


#model_structure = ft_net(class_num=1453, stride=2)
#model = load_network(model_structure)
model = torch.load('../model/ResNet18/net_org.pth')
model.classifier = nn.Sequential()
model.model.fc = nn.Sequential()
print('Model before quantization:')
print(model)

cpu_device = torch.device("cpu:0")
model.to(cpu_device)

# Make a copy of the model for layer fusion
fused_model = copy.deepcopy(model)

model.eval()
# The model has to be switched to evaluation mode before any layer fusion.
# Otherwise the quantization will not work correctly.
fused_model.eval()

# Fuse the model in place rather manually.

torch.quantization.fuse_modules(fused_model.model,
                                [["conv1", "bn1", "relu"]],
                                inplace=True)
for module_name, module in fused_model.model.named_children():
    if "layer" in module_name:
        for basic_block_name, basic_block in module.named_children():
            torch.quantization.fuse_modules(
                basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                inplace=True)
            for sub_block_name, sub_block in basic_block.named_children():
                if sub_block_name == "downsample":
                    torch.quantization.fuse_modules(sub_block,
                                                    [["0", "1"]],
                                                    inplace=True)


# Print FP32 model.
# Print fused model.
print("Fused")
print(fused_model)


quantized_model = QuantizedResNet18(model_fp32=fused_model)
# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.

# Using un-fused model will fail.
# Because there is no quantized layer implementation for a single batch normalization layer.
quantization_config = torch.quantization.get_default_qconfig("qnnpack")
# Custom quantization configurations
# quantization_config = torch.quantization.default_qconfig
# quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

quantized_model.qconfig = quantization_config


# https://pytorch.org/docs/master/torch.quantization.html#torch.quantization.prepare
torch.quantization.prepare(quantized_model, inplace=True)

# Use training data for calibration.
quantized_model.to(cpu_device)
quantized_model.eval()
for data in dataloaders['val']:
    img, label = data
    input_img = Variable(img)
    outputs = quantized_model(input_img)

quantized_model = torch.quantization.convert(quantized_model, inplace=True)
print("Quantized model:")
# print(quantized_model)
# quantized_model.model_fp32.classifier = model.classifier
# print("Quantized2:")
# quantized_model.model_fp32.classifier = nn.Sequential()
print(quantized_model)

quantized_model.eval()
print('Finish quantizing model.')

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=print_size_of_model(model,"fp32")
q=print_size_of_model(quantized_model,"int8")
print("{0:.2f} times smaller".format(f/q))

#################################################
model_dir = '../model/ResNet18_Qt'

quantized_model_filename = 'best_model_qnn.torchscript'
save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)


#################################################

'''

