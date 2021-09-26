from __future__ import print_function, division

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
from model import ft_net, ft_net_dense, ft_net_NAS, PCB
from random_erasing import RandomErasing
import scipy.io
import math
from shutil import copyfile

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ResNet18_Pruned', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Market-1501/pytorch',type=str, help='training dir path')
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



####################################################################################

####################################################################################
opt.nclasses = len(class_names)
name = 'ResNet18_Pruned'
file_num = '001'
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%file_num)
    print('Loading model: net_%s.pth'%file_num)
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

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

model_structure = ft_net(opt.nclasses, stride = opt.stride)
model_fp32 = load_network(model_structure)

model_fp32.classifier.classifier = nn.Sequential()
#print(model_fp32)

##### dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
print('Model_int8:')
print(model_int8)


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=print_size_of_model(model_fp32,"fp32")
q=print_size_of_model(model_int8,"int8")
print("{0:.2f} times smaller".format(f/q))

