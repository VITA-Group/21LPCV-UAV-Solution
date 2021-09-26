import sys
sys.path.insert(0, '..')
import os
import argparse

import torch
import torch.utils.data

import copy
import torch.nn as nn
from models.model import ft_net, ClassBlock
from models.pruned_model_half import ft_net_2
from models.pruned_model_quater import ft_net_4
from models.pruned_model import ft_net_8
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.pytorch.pruning import FPGMPruner
from nni.algorithms.compression.pytorch.pruning import SlimPruner
from nni.compression.pytorch.utils.counter import count_flops_params
######################################################################
add_new = True
my_file_name = 'net_final_erase_0.1.pth'
my_save_name = 'res_0.5.pth'
# Load model
#---------------------------

def load_network(network, save_path):
    # '../model/ResNet18/FPGM_0.5_finetuned.pth'
    print('Loading model: %s'%save_path)
    network.load_state_dict(torch.load(save_path), strict=True)
    return network


def get_out_channel(conv):
    weight_tensor = conv.weight
    out_list = []
    for i in range(weight_tensor.shape[0]):
        if sum(sum(sum(weight_tensor[i]))) != 0:
            out_list.append(i)
    return out_list

def set_weight_conv(last_c, out_c, ori_conv):
    bias = ori_conv.bias is not None
    conv = nn.Conv2d(len(last_c), len(out_c), kernel_size=ori_conv.kernel_size,
                     stride=ori_conv.stride, padding=ori_conv.padding, bias=bias)
    conv_shape = conv.weight.shape
    for o in range(conv_shape[0]):
        for i in range(conv_shape[1]):
            with torch.no_grad():
                conv.weight[o][i] = ori_conv.weight[out_c[o]][last_c[i]]
    if bias:
        for o in range(conv_shape[0]):
            conv.bias[o] = ori_conv.bias[out_c[o]]
    return conv

def set_weight_linear(last_c, out_c, ori_linear):
    bias = ori_linear.bias is not None
    linear = nn.Linear(len(last_c), out_c, bias=bias)
    linear_shape = linear.weight.shape
    for o in range(out_c):
        for i in range(linear_shape[1]):
            with torch.no_grad():
                linear.weight[o][i] = ori_linear.weight[o][last_c[i]]
    if bias:
        for o in range(linear_shape[0]):
            with torch.no_grad():
                linear.bias[o] = ori_linear.bias[o]
    return linear


def set_bn(out_c, ori_bn):
    bn = nn.BatchNorm2d(len(out_c), eps=ori_bn.eps, momentum=ori_bn.momentum, affine=ori_bn.affine,
                        track_running_stats=ori_bn.track_running_stats)
    bn.num_batches_tracked = ori_bn.num_batches_tracked
    for o in range(len(out_c)):
        with torch.no_grad():
            bn.weight[o] = ori_bn.weight[out_c[o]]
            bn.bias[o] = ori_bn.bias[out_c[o]]
            bn.running_mean[o] = ori_bn.running_mean[out_c[o]]
            bn.running_var[o] = ori_bn.running_var[out_c[o]]
    return bn


def save_network(network, save_path):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',folder_name,save_filename)
    torch.save(network.module.state_dict(), save_path)

##########################################################

# load model with structure: nclasses=3465, droprate=0.5, stride=2, circle=True
model_structure = ft_net(class_num=3465, droprate=0.5, stride=2, circle =True)
model = load_network(model_structure, '../model/ResNet18/' + my_file_name)
#model = torch.load('../model/ResNet18/net_org.pth')
print(model)

device = torch.device("cpu")
dummy_input = torch.randn([256, 3, 256, 128]).to(device)
sparsity = '0.5'
config_list = [{ 'sparsity': float(sparsity), 'op_types': ['Conv2d'] }]
pruner = FPGMPruner(model, config_list, dependency_aware=True, dummy_input=dummy_input)
model = pruner.compress()
pruner.get_pruned_weights()
save_filename = 'FPGM_0.5.pth'
model_path='../model/ResNet18_Pruned/' + save_filename                  # ex: 'model/ResNet18_Pruned/FPGM_0.5.pth'
mask_path = '../model/ResNet18_Pruned/' + 'mask_' + save_filename         # ex: 'model/ResNet18_Pruned/FPGM_net_0.5.pth'
pruner.export_model(model_path=model_path, mask_path=mask_path)
############################################################
model_structure = ft_net(class_num=3465, droprate=0.5, stride=2, circle =True)
model = load_network(model_structure, '../model/ResNet18_Pruned/' + save_filename)
print(model)
###########################################################
model = model.eval()
model_new = ft_net(class_num=3465, droprate=0.5, stride=2, circle =True)

model_new = model.eval()
model_new = copy.deepcopy(model)
new_dict = {}

model_list = list(model.model._modules.items())
last_channels = [0, 1, 2]

# Delete channels for Conv1
for i in range(4):
    module = model_list[i][1]
    #print(module)
    if isinstance(module, torch.nn.Conv2d):
        out_channels = get_out_channel(module)
        new = set_weight_conv(last_channels, out_channels, module)
        last_channels = out_channels
        new_dict[model_list[i][0]] = new
    if isinstance(module, torch.nn.BatchNorm2d):
        new = set_bn(out_channels, module)
        new_dict[model_list[i][0]] = new
model_new.model.conv1 = new_dict['conv1']
model_new.model.bn1 = new_dict['bn1']



# Delete channels for layer1
model_list = list(model.model.layer1._modules.items())
new_dict_layer1 = {}
for i in range(len(model_list)):
    module = model_list[i][1]
    new_dict_layer1[i] = {}
    module_l = list(module._modules.items())

    for j in range(len(module_l)):
        module_b = module_l[j][1]
        if isinstance(module_b, torch.nn.Conv2d):
            out_channels = get_out_channel(module_b)
            new = set_weight_conv(last_channels, out_channels, module_b)
            last_channels = out_channels
            new_dict_layer1[i][module_l[j][0]] = new
        if isinstance(module_b, torch.nn.BatchNorm2d):
            new = set_bn(out_channels, module_b)
            new_dict_layer1[i][module_l[j][0]] = new

for idx, m in enumerate(model_new.model.layer1.children()):
    conv_dict = new_dict_layer1[idx]
    m.conv1 = conv_dict['conv1']
    m.bn1 = conv_dict['bn1']
    m.conv2 = conv_dict['conv2']
    m.bn2 = conv_dict['bn2']


# Code for layer2 ~ layer4
def copy_res_layer(last_channels, model_l):
    last_down_channels = last_channels
    new_dict = {}
    new_dict['downsample'] = {}
    for i in range(len(model_l)):
        module = model_l[i][1]
        new_dict[i] = {}
        module_l = list(module._modules.items())

        for j in range(len(module_l)):
            module_b = module_l[j][1]
            if isinstance(module_b, torch.nn.Conv2d):
                out_channels = get_out_channel(module_b)
                new = set_weight_conv(last_channels, out_channels, module_b)
                last_channels = out_channels
                new_dict[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.BatchNorm2d):
                new = set_bn(out_channels, module_b)
                new_dict[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.ReLU):
                continue
            else:
                module_down = list(module_b._modules.items())
                for k in range(len(module_down)):
                    module_k = module_down[k][1]
                    if isinstance(module_k, torch.nn.Conv2d):
                        out_channels = get_out_channel(module_k)
                        new = set_weight_conv(last_down_channels, out_channels, module_k)
                        last_channels = out_channels
                        new_dict['downsample'][module_down[k][0]] = new
                    elif isinstance(module_k, torch.nn.BatchNorm2d):
                        new = set_bn(out_channels, module_k)
                        new_dict['downsample'][module_down[k][0]] = new

    return last_channels, new_dict


def set_values_res_layer(layer, new_dict):
    for idx, m in enumerate(layer.children()):
        conv_dict = new_dict[idx]
        m.conv1 = conv_dict['conv1']
        m.bn1 = conv_dict['bn1']
        m.conv2 = conv_dict['conv2']
        m.bn2 = conv_dict['bn2']
        if idx == 0:
            m.downsample = nn.Sequential(
                new_dict['downsample']['0'],
                new_dict['downsample']['1'], )

last_down_channels_1 = last_channels
last_channels, new_dict = copy_res_layer(last_channels, list(model.model.layer2._modules.items()))
set_values_res_layer(model_new.model.layer2, new_dict)

last_down_channels_2 = last_channels
last_channels, new_dict = copy_res_layer(last_channels, list(model.model.layer3._modules.items()))
set_values_res_layer(model_new.model.layer3, new_dict)

last_down_channels_3 = last_channels
last_channels, new_dict = copy_res_layer(last_channels, list(model.model.layer4._modules.items()))
set_values_res_layer(model_new.model.layer4, new_dict)
last_down_channels_4 = last_channels

model_new.classifier = model.classifier
if add_new:
    print("Initialize new classifier!")
    model_new.classifier = ClassBlock(256, 3465, droprate=0.5, return_f=True)
else:
    print("Using previously pruned classifier!")
    model_new.classifier.add_block[0] = set_weight_linear(last_channels, 512, model.classifier.add_block[0])

# print(model_new.classifier.add_block[0].weight.size())


# downsize classifier
#model_structure = ft_net(class_num=1453,stride=2)
#model_new.classifier = ClassBlock(256, 3465, droprate=0.5, return_f=True)


print(model_new)
device = torch.device("cpu")
dummy_input = torch.randn([256, 3, 256, 128]).to(device)
flops, params, results = count_flops_params(model_new, dummy_input)
print(f"FLOPs: {flops}, params: {params}")
torch.save(model_new.state_dict(), '../model/ResNet18_Pruned/' + my_save_name)
print("Saving as:", my_save_name)

if add_new:
    print("Initialize new classifier!")
else:
    print("Using previously pruned classifier!")