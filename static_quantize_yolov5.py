#Static quantization module for YOLOv5, use this rename_dict function to quantize original YOLOv5
import argparse
import tqdm
import torch
import torch.utils.data
from collections import OrderedDict
from models.quanted_yolov5 import YoloV5_quanted
import copy
import torch.quantization._numeric_suite as ns
from utils.general import *
import os
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate-folder', type=str, default='../data/9p6b_save/images',
                        help='Path to folder with images for calibrating')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Exported pruned model name in save dir')
    parser.add_argument('--backends', type=str, default='fbgemm', help='Quantization backends')
    args = parser.parse_args()

    images_folder = args.calibrate_folder
    pbar = tqdm.tqdm(os.listdir(images_folder), desc='Test', ncols=80)

    # Get model architecture from yaml
    anchors = [[20,26, 32,60, 66,46]]
    #anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    device = 'cpu'
    # Here must match the model architecture
    net = YoloV5_quanted(anchors=anchors).to(device)
    ckpt_dir = args.pretrain_model
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    # state_dict = ckpt
    state_dict = ckpt['model'].state_dict()
    new_state_dict = rename_state_dict_keys(state_dict)
    net.load_state_dict(new_state_dict, strict=True)

    # fp_net for floating point reference
    fp_net = copy.deepcopy(net)

    # Must set eval for Static Post Quantization
    net.eval()
    fp_net.eval()

    # Get quantization configure for post training quantization
    qcf = torch.quantization.get_default_qconfig(args.backends)

    # Set quantization configure
    net.qconfig = qcf

    # Fuse all parts in the FOTs model
    net.fuse_model()

    # Prepare the quantization for all parts
    torch.quantization.prepare(net, inplace=True)

    # Use some images from the folder for evaluation or calibration
    for image_name in pbar:
        image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (640, 640))
        image = torch.tensor(image)
        image = image.view(1, 3, 640, 640)
        image = image.to(device, non_blocking=True).float() / 255.0
        pred = net(image)

    # Convert all parts into quantized ones
    quantized_net = torch.quantization.convert(net, inplace=True)
    print(quantized_net)
    wt_compare_dict = ns.compare_weights(fp_net.state_dict(), quantized_net.state_dict())

    for key in wt_compare_dict:
        print(key, compute_error(wt_compare_dict[key]['float'], wt_compare_dict[key]['quantized'].dequantize()))

    # Get the backend and save the torchscript files to the specific path
    backend = args.backends
    # torch.onnx.export(net,
    #                   image,
    #                   '{}/yolov5s_{}.onnx'.format(args.save_dir, backend),
    #                   export_params=True,
    #                   opset_version=11,
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {0: 'batch_size'},
    #                                 'output': {0: 'batch_size'}})
    traced_module = torch.jit.trace(net, image, strict=True)
    torch.jit.save(traced_module, '{}/yolov5s_{}.torchscript'.format(args.save_dir, backend))
