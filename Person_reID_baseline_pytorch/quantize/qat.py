# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
sys.path.insert(0, '..')


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
#from model import ft_net, ft_net_dense, ft_net_NAS, PCB
from models.pruned_model_quater import ft_net_4
from models.pruned_model import ft_net_8
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
from circle_loss import CircleLoss, convert_label_to_similarity
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
parser.add_argument('--name',default='ResNet18_Pruned', type=str, help='output model name')
parser.add_argument('--data_dir',default='../../Multi-2/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_false', help='use all training data' )
parser.add_argument('--color_jitter', action='store_false', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
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
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#
resize_w, resize_h = 128, 64
print("Resize: ", resize_w, resize_h)
transform_train_list = [
    #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((resize_w, resize_h), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((resize_w, resize_h)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(resize_w, resize_h),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


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

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)
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

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_one_epoch(model, criterion, optimizer, device, ntrain_batches):
    since = time.time()
    model.train()
    #model.to(device)
    cnt = 0
    running_loss = 0.0
    running_corrects = 0.0
    for data in dataloaders['train']:
        cnt += 1
        # get the inputs
        inputs, labels = data
        now_batch_size,c,h,w = inputs.shape
        print(inputs.shape)
        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
            running_loss += loss.item() * now_batch_size
        else :  # for the old version like 0.3.0 and 0.3.1
            running_loss += loss.data[0] * now_batch_size
        running_corrects += float(torch.sum(preds == labels.data))

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']

        if cnt >= ntrain_batches:
            print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
            # time
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()
        return model

    


######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

######################################################################
# Load model
#---------------------------

def load_network(network):
    name = 'ResNet18'
    file_name = 'net_resize_128_finetuned.pth'
    save_path = os.path.join('../model' , name, file_name)
    print('Loading model from: ', save_path)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Load a pretrainied model and reset final fully connected layer.
opt.nclasses = len(class_names)
model_structure = ft_net_8(opt.nclasses, stride = opt.stride)
model = load_network(model_structure)
model.model.fc = nn.Sequential()
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
# functions for QAT
######################################################################
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
######################################################################
# Train and evaluate
name = opt.name
#dir_name = os.path.join('./model',name)
#if not os.path.isdir(dir_name):
 #   os.mkdir(dir_name)
#record every run
#copyfile('./train.py', dir_name+'/train.py')
#copyfile('./model.py', dir_name+'/model.py')

# save opts


# copy and get fused model

fused_model = copy.deepcopy(model)
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
qat_model = QuantizedResNet18(model_fp32=fused_model)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
torch.quantization.prepare_qat(qat_model, inplace=True)
#######################################################
# start training a quantized model
cpu_device = torch.device("cpu")
gpu_device = torch.device("cuda:0")
num_train_batches = 20
criterion = nn.CrossEntropyLoss()
optimizer = optimizer_ft
# data_loader = dataloaders
# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch

for nepoch in range(30):
    train_one_epoch(qat_model, criterion, optimizer, gpu_device, num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
#print(qat_model)
qat_model.eval()
qat_model.model_fp32.classifier = nn.Sequential()
quantized_model = torch.quantization.convert(qat_model, inplace=False)


print(quantized_model)
#quantized_model.model_fp32.classifier.classifier = nn.Sequential()
########################################################
# save model to torchscript format

def save_torchscript_model(model, model_dir, model_filename):
    device = torch.device("cpu")
    dummy_input = torch.randn([256, 3, resize_w, resize_h]).to(device)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.trace(model, dummy_input, strict=True), model_filepath) 

model_dir = '../model/ResNet18_Qt'

quantized_model_filename = 'resize_128_qnn_A30.torchscript'
save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)




