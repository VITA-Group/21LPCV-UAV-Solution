# -*- coding: utf-8 -*-
# Test normal (unpruned, unquantized model)
from __future__ import print_function, division


import sys

print(sys.executable)

import sys

print(sys.executable)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from models.model import ft_net
from models.pruned_model_half import ft_net_2
from models.pruned_model_quater import ft_net_4
from models.pruned_model import ft_net_8
from nni.compression.pytorch.utils.counter import count_flops_params
# torch.backends.cudnn.enabled=False

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

# dataset 1: Combined_Dataset
# dataset 2: Market-1501
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market-1501/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ResNet18_Pruned', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_false', help='use densenet121')
parser.add_argument('--PCB', action='store_false', help='use PCB' )
parser.add_argument('--multi', action='store_false', help='use multiple query' )
parser.add_argument('--fp16', action='store_false', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

resize_w, resize_h = 256 // 2 , 128  // 2
opt = parser.parse_args()
fp16 = False
opt.fp16 = False
opt.circle = False
opt.PCB = False
opt.use_NAS = False
opt.use_dense = False
opt.color_jitter = False
opt.tran_all = False
opt.multi = False
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = [0]



print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
print(gpu_ids)

my_file_path = 'ResNet18_Pruned'
my_checkpoint = 'Erase_0.875_cut_64_64_lr1.pth'
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((resize_w,resize_h), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop  ##########################################      
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
# use_gpu = torch.cuda.is_available()
use_gpu = True
######################################################################
# Load model
#---------------------------

def load_network(network):
    save_path = os.path.join('./model', my_file_path, my_checkpoint)
    print('Loading model from: ', save_path)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,64).zero_().cuda()
        #ff = torch.FloatTensor(n,512).zero_()
        for i in range(1):
            if(i==1):
                img = fliplr(img)
            input_img = img.cuda()
            # input_img = Variable(img)
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)



######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    model_structure = ft_net_8(3465, stride = opt.stride)


if opt.PCB:
    model_structure = PCB(opt.nclasses)


# load pruned / unpruned model

print('nclasses:')
print(opt.nclasses)
print('stride:')
print(opt.stride)

#model = torch.load('./model/ResNet18/net_org.pth')
model = load_network(model_structure)
# load quantized model
'''quantized_model_filepath = './model/ResNet18_Qt/quantized_net_001.pt'
cpu_device = torch.device("cpu:0")
model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)'''




# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()
print(model)
#test number of FLOPs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn([256, 3, resize_w, resize_h]).to(device)
flops, params, results = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params}")

# Change to test mode
model = model.eval()
model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])
    if opt.multi:
        mquery_feature = extract_feature(model,dataloaders['multi-query'])
    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.name)
result = './model/%s/result.txt'%opt.name
os.system('/home/wuzhenyu_sjtu/XinHu/anaconda3/envs/lpcv/bin/python evaluate_gpu.py | tee -a %s'%result)

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)

