# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
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
from models.model import ft_net, ft_net_dense, ft_net_NAS, PCB
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
from circle_loss import CircleLoss, convert_label_to_similarity
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.pytorch.pruning import AGPPruner
from nni.algorithms.compression.pytorch.pruning import SimulatedAnnealingPruner, ADMMPruner, NetAdaptPruner, AutoCompressPruner, SlimPruner
from nni.algorithms.compression.pytorch.pruning import AMCPruner
import copy


version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ResNet18', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Multi-2/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--parallel', action='store_true', help='use data parallel' )
opt = parser.parse_args()


data_dir = opt.data_dir
folder_name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = [0]
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

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

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)

pruner_type = 'Slim'
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.



criterion_circle = CircleLoss(m=0.25, gamma=32)
def train_model(model, criterion, optimizer, scheduler, epoch):
    #scheduler.step()
    model.train()
    #model.to(device)
    running_loss = 0.0
    running_corrects = 0.0
    since = time.time()
    for data in dataloaders['train']:
        inputs, labels = data
        now_batch_size,c,h,w = inputs.shape
        if now_batch_size<opt.batchsize: # skip the last batch
            continue
        if use_gpu:
            inputs = Variable(inputs.cuda().detach())
            labels = Variable(labels.cuda().detach())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)

        logits, ff = outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        loss = criterion(logits, labels) + criterion_circle(*convert_label_to_similarity( ff, labels))/now_batch_size
        _, preds = torch.max(logits.data, 1)

        loss.backward()
        optimizer.step()
        # statistics
        if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
            running_loss += loss.item() * now_batch_size
        else :  # for the old version like 0.3.0 and 0.3.1
            running_loss += loss.data[0] * now_batch_size
        running_corrects += float(torch.sum(preds == labels.data))
    scheduler.step()
    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects / dataset_sizes['train']
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print()
    print('Train Epoch: {}'.format(epoch))
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))
    return model


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    since = time.time()
    for data in dataloaders['val']:
        inputs, labels = data
        now_batch_size,c,h,w = inputs.shape
        if now_batch_size<opt.batchsize: # skip the last batch
            continue
        if use_gpu:
            inputs = Variable(inputs.cuda().detach())
            labels = Variable(labels.cuda().detach())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)
        logits, ff = outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        loss = criterion(logits, labels) + criterion_circle(*convert_label_to_similarity( ff, labels))/now_batch_size
        _, preds = torch.max(logits.data, 1)

         # statistics
        if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
            running_loss += loss.item() * now_batch_size
        else :  # for the old version like 0.3.0 and 0.3.1
            running_loss += loss.data[0] * now_batch_size
        running_corrects += float(torch.sum(preds == labels.data))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print()
    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects / dataset_sizes['val']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val', epoch_loss, epoch_acc))

    return epoch_acc

######################################################################
# Save model



######################################################################
# Load model
#---------------------------
name = 'ResNet18'
checkpoint = 'net_org_4.pth'
def load_network(network):
    save_path = os.path.join('./model',name,checkpoint)
    print('Loading model: ', checkpoint)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
opt.nclasses = len(class_names)
#model_structure = ft_net(opt.nclasses, opt.droprate, stride = opt.stride, circle=True)
#model = load_network(model_structure)
model = torch.load('./model/ResNet18/net_org.pth')
print(model)

ignored_params = list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1*opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)


# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
name = opt.name
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./models/model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

print(model)
# model to gpu
model = model.cuda()
criterion = nn.CrossEntropyLoss()


def trainer(model, optimizer, criterion, epoch):
    return train_model(model, criterion, optimizer_ft, exp_lr_scheduler, epoch)

def evaluator(model):
    return test_model(model, criterion)

def short_term_fine_tuner(model, epochs=1):
    for epoch in range(epochs):
        train_model(model, criterion, optimizer_ft, exp_lr_scheduler, epoch)


# configuration for pruner
sparsity = '0.75'
config_list = [{
            'sparsity': float(sparsity),
            'op_types': ['BatchNorm2d']
        }]
print(config_list)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn([256, 3, 256, 128]).to(device)

if pruner_type == 'Auto':
    pruner = AutoCompressPruner(
    model, config_list, trainer=trainer, evaluator=evaluator, dummy_input=dummy_input,
    num_iterations=1, optimize_mode='maximize', base_algo='fpgm',
    cool_down_rate=0.7, admm_num_iterations=1, admm_epochs_per_iteration=1,
    experiment_data_dir='./exp')
elif pruner_type == 'NetAdapt':
    pruner = NetAdaptPruner(model, config_list, short_term_fine_tuner=short_term_fine_tuner, evaluator=evaluator, base_algo='l1', experiment_data_dir='./exp')
elif pruner_type == 'Simulated':
    pruner = SimulatedAnnealingPruner(model, config_list, evaluator=evaluator, base_algo='l1', cool_down_rate=0.9, experiment_data_dir='./')
elif pruner_type == 'Slim':
    pruner = SlimPruner(model, config_list, optimizer_ft, trainer, criterion)

model = pruner.compress()
print(model)
pruner.get_pruned_weights()

#model.classifier = nn.Sequential()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn([256, 3, 256, 128]).to(device)
flops, params, results = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params}")
save_filename = 'Slim_0.75.pth'
model_path='model/ResNet18_Pruned/' + save_filename                  # ex: 'model/ResNet18_Pruned/FPGM_0.5.pth'
mask_path = 'model/ResNet18_Pruned/' + 'mask_' + save_filename         # ex: 'model/ResNet18_Pruned/FPGM_net_0.5.pth'
pruner.export_model(model_path=model_path, mask_path=mask_path)
