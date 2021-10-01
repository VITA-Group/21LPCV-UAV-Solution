# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import logging
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
from models.model import ft_net
from models.pruned_model_half import ft_net_2
from models.pruned_model_quater import ft_net_4
from models.pruned_model import ft_net_8
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
from circle_loss import CircleLoss, convert_label_to_similarity
from nni.algorithms.compression.pytorch.pruning import FPGMPruner
from nni.algorithms.compression.pytorch.pruning import SlimPruner
from nni.compression.pytorch.utils.counter import count_flops_params


version =  torch.__version__
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ResNet18_Pruned', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Multi-5/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_false', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_false', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_false', help='use PCB+ResNet50' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--fp16', action='store_false', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

opt = parser.parse_args()

fp16 = False
opt.fp16 = False
opt.PCB = False
opt.use_NAS = False
opt.use_dense = False
opt.color_jitter = False
opt.tran_all = False
data_dir = opt.data_dir
folder_name = opt.name
mul = 0.5
resize_w, resize_h = int(256 * mul), int(128 * mul)
gpu_ids = [7]
torch.cuda.set_device(gpu_ids[0])
#cudnn.benchmark = True
my_file_path = 'ResNet18_Pruned'
my_checkpoint = 'Erase_0.875_cut_64.pth'

######################################################################
# Load Data
# ---------
#

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




def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32) # gamma = 64 may lead to a better result.
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                '''if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)'''
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if opt.circle:
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels) + criterion_circle(*convert_label_to_similarity( ff, labels))/now_batch_size
                    #loss = criterion_circle(*convert_label_to_similarity( ff, labels))
                    _, preds = torch.max(logits.data, 1)

                else:  #  norm
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                

                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss*warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 5 or epoch%10 == 1:
                    save_network(model, epoch)
                #draw_curve(epoch)
            if phase == 'train':
                scheduler.step()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, epoch)
    #save_filename = 'net_%s.pth'% epoch
    #torch.save(model, 'Auto_final.pth')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend() 
        ax1.legend()
    fig.savefig( os.path.join('./model',folder_name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = my_checkpoint[:-4] + '_%s.pth'% epoch_label
    save_path = os.path.join('./model',folder_name,save_filename)
    print(save_path)
    torch.save(network.state_dict(), save_path)
    #if torch.cuda.is_available():
        #network.cuda(0)

######################################################################
# Load model
#---------------------------

def load_network(network):
    save_path = os.path.join('./model', my_file_path, my_checkpoint)
    print('Loading model from: ', save_path)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#


opt.nclasses = len(class_names)

###
#model 
model_structure = ft_net_8(opt.nclasses, opt.droprate, stride = opt.stride, circle=True)

model = load_network(model_structure)
#model = torch.load('./model/ResNet18/net_org.pth')
##model.classifier = ClassBlock(64, class_num=2424, droprate=0.5)
#print('Class num:', opt.nclasses)
#print(model)
model = model.cuda()





#####################################################################

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
#copyfile('./train.py', dir_name+'/train.py')
#copyfile('./models/model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)



criterion = nn.CrossEntropyLoss()




model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=65)
### pruning
#print('start finetuning...')
#model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=60)

'''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn([256, 3, 256, 128]).to(device)
sparsity = '0.75'
config_list = [{ 'sparsity': float(sparsity), 'op_types': ['BatchNorm2d'] }]
pruner = SlimPruner(model, config_list)                             #dependency_aware=True, dummy_input=dummy_input
model = pruner.compress()
pruner.get_pruned_weights()
print(model)
save_filename = 'Slim_0.75.pth'
model_path='model/ResNet18_Pruned/' + save_filename                  # ex: 'model/ResNet18_Pruned/FPGM_0.5.pth'
mask_path = 'model/ResNet18_Pruned/' + 'mask_' + save_filename         # ex: 'model/ResNet18_Pruned/FPGM_net_0.5.pth'
pruner.export_model(model_path=model_path, mask_path=mask_path)'''

#'op_names': ['model.conv1', 'model.layer1.0.conv1', 'model.layer1.0.conv2', 'model.layer1.1.conv1', 'model.layer1.1.conv2']







