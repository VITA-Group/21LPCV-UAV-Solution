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
from models.pruned_model import ft_net_8
from random_erasing import RandomErasing
import scipy.io
import math
from nni.compression.pytorch.utils.counter import count_flops_params
from shutil import copyfile
# torch.backends.quantized.engine = 'qnnpack'
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ResNet18_Qt', type=str, help='output model name')
parser.add_argument('--data_dir',default='../eval-hard-new/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_false', help='use all training data' )
parser.add_argument('--color_jitter', action='store_false', help='use color jitter in training' )
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
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
parser.add_argument('--multi', action='store_true', help='use multiple query' )
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
gpu_ids = [4]
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
###################################################
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
# f=print_size_of_model(model,"fp32")
# q=print_size_of_model(quantized_model,"int8")
# print("{0:.2f} times smaller".format(f/q))

#################################################

# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
w, h = 128, 64
data_transforms = transforms.Compose([
    transforms.Resize((w,h), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])





image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
opt.nclasses = len(class_names)

###

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#

def fliplr(img):
    #flip horizontal
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):

    features = torch.FloatTensor()
    count = 0
    since = time.time()
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        # c, h, w = 3, 256, 128
        count += n
        print(count)

        ff = torch.FloatTensor(n,64).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        since = time.time()
        for i in range(1):
            if (i==1):
                img = fliplr(img)
            input_img = Variable(img)
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)

                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff.data.cpu()), 0)
        time_elapsed = time.time() - since
        print(time_elapsed)
        '''if count % 100 == 0:
            time_elapsed = time.time() - since
            print('Feature extracted in %f seconds'%time_elapsed)
            print
            since = time.time()'''
    print(features)
    print(features.size())
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
print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))



# load quantized model
######################################################################
# Load Collected data Trained model

def load_network(network):
    name = 'ResNet18_Qt'
    file_name = 'best_model.pt'
    save_path = os.path.join('./model',name,file_name)
    print('Loading model: ', file_name)
    network.load_state_dict(torch.load(save_path))
    return network

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


# load quantized model
cpu_device = torch.device("cpu:0")
# model1 = load_torchscript_model(model_filepath='./model/ResNet18_Qt/qat_net_59.pt', device=cpu_device)
model1 = load_torchscript_model(model_filepath='./model/ResNet18_Qt/resize_128_qnn_A29.torchscript', device=cpu_device)
print(model1)

# load original model
# model_structure = ft_net(class_num=1453, stride=2)
# model2 = load_network(model_structure)

# Remove the final fc layer and classifier layer


# Change to test mode
model = model1
model = model.eval()
#if use_gpu:cd 
    #model = model.cuda()

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
os.system('/home/grads/l/leomiao/anaconda3/envs/lpcv/bin/python evaluate_gpu.py | tee -a %s'%result)

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)






