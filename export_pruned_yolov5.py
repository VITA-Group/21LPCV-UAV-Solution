# Functions for mask pruned model
import sys
import os
import argparse
from models.yolo import parse_model
from copy import deepcopy
import torch
import torch.utils.data
from collections import OrderedDict
import torch.nn as nn
import yaml
from pathlib import Path


"""
This code aims to delete the sparse channels for structured pruning obtained from prune_yolov5.py
"""


# Export output channels by deleting sparse channels
# Input: layer: input layer; input_channels: output non_sparse channels list for previous layer
# Output: exported non_sparse layer, output non_sparse channels list for current layer
def get_output_channels(layer, input_channels):
    output_channels = []
    new_v = []
    current_channel = 0
    for channel in layer:
        if torch.norm(channel).item() > 0:
            new_v.append((channel[input_channels].numpy()))
            output_channels.append(current_channel)
        current_channel += 1

    return torch.as_tensor(new_v), output_channels


# Export output channels by deleting sparse channels for the last layer
# Input: layer: input layer; input_channels: output non_sparse channels list for previous layer
# Output: exported non_sparse layer, output non_sparse channels list for current layer
def get_last_output_channels(layer, input_channels):
    output_channels = []
    new_v = []
    current_channel = 0
    for channel in layer:
        new_v.append((channel[input_channels].numpy()))
        output_channels.append(current_channel)
        current_channel += 1

    return torch.as_tensor(new_v), output_channels


# rename the state_dict and keys to eliminate sparse channels in pruned model
def rename_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    input_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    output_channels = []
    for k, v in state_dict.items():
        if k[:5] == 'model':
            name = k[6:]
            print(name)
            if name[0] == '0' or name[0:2] == '1.' or name[0] == '3' or name[0] == '5' or name[0] == '7' or \
                    name[0:2] == '10' or name[0:2] == '14' or name[0:2] == '18' or name[0:2] == '21':   # Conv
                if k[-11:] == 'conv.weight':
                    if name[0] != '0' and name[0:2] != '1.':
                        input_channels = cv3_output_channels
                    new_state_dict[name], output_channels = get_output_channels(v, input_channels)
                    input_channels = output_channels
                    if name[0:2] == '14':
                        conv_14_output_channels = input_channels
                    if name[0:2] == '10':
                        conv_10_output_channels = input_channels
                    # tmp_state_dict = tmp_state_dict[:, input_channels]
                elif k[-9:] == 'bn.weight' or k[-7:] == 'bn.bias' or k[-12:] == 'running_mean' or \
                        k[-11:] == 'running_var':
                    new_state_dict[name] = v[input_channels]
                else:  # for bn.num_batches_tracked
                    new_state_dict[name] = v
                    # input_channels = output_channels

            elif name[0:2] == '2.' or name[0] == '4' or name[0] == '6' or name[0] == '9' or name[0:2] == '13' or \
                name[0:2] == '17' or name[0:2] == '20' or name[0:2] == '23':  # C3
                # print(name[0:2])
                if k[-11:] == 'conv.weight':
                    # print(name[0:2])
                    if name[2:5] == 'cv1' or name[3:6] == 'cv1':
                        if name[0:2] == '13':
                            input_channels = input_channels + cv3_6_output_channels
                        if name[0:2] == '17':
                            input_channels = input_channels + cv3_4_output_channels
                        if name[0:2] == '20':
                            input_channels = input_channels + conv_14_output_channels
                        if name[0:2] == '23':
                            input_channels = input_channels + conv_10_output_channels
                        cv2_input_channels = input_channels
                        new_state_dict[name], output_channels = get_output_channels(v, input_channels)
                        input_channels = output_channels
                        cv1_input_channels = output_channels
                    elif name[2:5] == 'cv2' or name[3:6] == 'cv2':
                        new_state_dict[name], output_channels = get_output_channels(v, cv2_input_channels)
                        cv2_input_channels = output_channels
                        input_channels = output_channels
                    elif name[2:5] == 'cv3' or name[3:6] == 'cv3':
                        input_channels = input_channels + cv2_input_channels
                        new_state_dict[name], output_channels = get_output_channels(v, input_channels)
                        input_channels = output_channels
                        cv3_output_channels = input_channels
                        if name[0] == '4':
                            cv3_4_output_channels = cv3_output_channels
                        if name[0] == '6':
                            cv3_6_output_channels = cv3_output_channels
                        if name[0:2] == '17':
                            cv3_17_output_channels = cv3_output_channels
                        if name[0:2] == '20':
                            cv3_20_output_channels = cv3_output_channels
                        if name[0:2] == '23':
                            cv3_23_output_channels = cv3_output_channels

                    elif name[2] == 'm' or name[3] == 'm':
                        if name[2:9] == 'm.0.cv1' or name[3:10] == 'm.0.cv1':
                            input_channels = cv1_input_channels
                        new_state_dict[name], output_channels = get_output_channels(v, input_channels)
                        input_channels = output_channels

                elif k[-9:] == 'bn.weight' or k[-7:] == 'bn.bias' or k[-12:] == 'running_mean' or \
                        k[-11:] == 'running_var':
                    new_state_dict[name] = v[output_channels]
                else:  # for bn.num_batches_tracked
                    new_state_dict[name] = v

            elif name[0] == '8':    # SPP
                if k[-11:] == 'conv.weight':
                    if name[2:5] == 'cv2':
                        input_channels = input_channels + input_channels + input_channels + input_channels
                    new_state_dict[name], output_channels = get_output_channels(v, input_channels)
                    input_channels = output_channels
                elif k[-9:] == 'bn.weight' or k[-7:] == 'bn.bias' or k[-12:] == 'running_mean' or \
                        k[-11:] == 'running_var':
                    new_state_dict[name] = v[output_channels]
                else:  # for bn.num_batches_tracked
                    new_state_dict[name] = v

            elif name[0:2] == '24':   # Detect
                if name[3:13] == 'm.0.weight':
                    new_state_dict[name], output_channels = get_last_output_channels(v, cv3_17_output_channels)
                    # input_channels = output_channels
                elif name[3:13] == 'm.1.weight':
                    new_state_dict[name], output_channels = get_last_output_channels(v, cv3_20_output_channels)
                    # input_channels = output_channels
                elif name[3:13] == 'm.2.weight':
                    new_state_dict[name], output_channels = get_last_output_channels(v, cv3_23_output_channels)
                    # input_channels = output_channels
                elif name[-4:] == 'bias':
                    print(len(output_channels))
                    new_state_dict[name] = v[output_channels]
                else:  # for bn.num_batches_tracked
                    new_state_dict[name] = v
            else:
                print('Aloha')
                raise NameError('Wrong Architecture')

    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--prune-model', type=str, default=None, help='Pretrained model name in save dir')
    args = parser.parse_args()

    # --------------- Change model architecture here for different sparsity rate --------------------
    # Get model architecture from yaml
    cfg_dir = "models/yolov5s.yaml"
    yaml_file = Path(cfg_dir).name
    with open(cfg_dir) as f:
        yaml = yaml.safe_load(f)
    ch = 3
    ch = yaml['ch'] = yaml.get('ch', ch)
    yolov5, save = parse_model(deepcopy(yaml), ch=[ch])
    model = yolov5.to(torch.device("cpu"))
    model = model.eval()
    # --------------- Change model architecture here for different sparsity rate --------------------

    ckpt = args.prune_model
    state_dict = torch.load(ckpt, map_location='cpu')
    new_state_dict = rename_state_dict_keys(state_dict)
    for k, v in new_state_dict.items():
        print(k)
        print(v.shape)
    model.load_state_dict(new_state_dict, strict=True)

    # Save Pruned Yolov5 model
    torch.save(model.state_dict(), '{}/exported_pruned_model.pth'.format(args.save_dir))
