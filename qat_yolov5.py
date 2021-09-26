# QAT module for YOLOv5
import argparse
import pdb
import numpy as np
import tqdm
import torch
import torch.utils.data
from qat_train import train
import yaml
from pathlib import Path
from models.quanted_yolov5 import YoloV5_quanted
import copy
import torch.quantization._numeric_suite as ns
import os
import cv2
import torch.distributed as dist
from utils.torch_utils import select_device
#from static_quantize_yolov5 import rename_state_dict_keys, compute_error
from utils.general import *
from test_qat import *



torch.backends.quantized.engine = 'qnnpack'


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1, 3, 299, 299)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        #print(model_1(x))

        y1 = model_1(x)[0].detach().cpu().numpy()

        y2 = model_2(x)[0].detach().cpu().numpy()
        if not np.allclose(a=y1, b=y2, rtol=rtol, atol=atol,
                           equal_nan=False):
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate-folder', type=str, default='../data/9p6b_save/images', help='Path to folder with images for calibrating')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--backends', type=str, default='fbgemm', help='Quantization backends')
    parser.add_argument('--weights', type=str, default='weights/trained_yolov5s.pt', help='path of weights to prune')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/vip_new.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--epochs_each_experiment', type=int, default=1, help='epochs for each experiment')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    args = parser.parse_args()


    print(args)
    # Set DDP variables
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(args.global_rank)
    if args.global_rank in [-1, 0]:
        #check_git_status()
        check_requirements(exclude=('pycocotools', 'thop'))

    images_folder = args.calibrate_folder
    pbar = tqdm.tqdm(os.listdir(images_folder), desc='Test', ncols=80)

    # Get model architecture from yaml
    anchors = [[20,26, 32,60, 66,46]]
    # anchors = [[10,13, 16,30, 33,23]]
    # anchors = [[10,13, 16,30, 33,23], 
    #            [30,61, 62,45, 59,119],
    #            [116,90, 156,198, 373,326]]

    """
    [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
    """
    # Device
    cpu_device = torch.device("cpu:0")
    # gpu_device = torch.device("cuda:0,1,2,3,4,5,6,7")

    net = YoloV5_quanted(anchors=anchors).to(cpu_device)
    ckpt_dir = args.weights
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    state_dict = ckpt['model'].state_dict()
    new_state_dict = rename_state_dict_keys(state_dict)
    net.load_state_dict(new_state_dict, strict=True)  # Need to rename

    net.to(cpu_device)

    # fp_net for floating point reference
    fp_net = copy.deepcopy(net)

    # Must set train for Quantization Aware Training
    net.train()
    fp_net.train()
    # Fuse all parts in the FOTs model
    net.fuse_model()

    net.eval()
    fp_net.eval()
    # Check model equivalent for model and fused model
    assert model_equivalence(
        model_1=net,
        model_2=fp_net,
        device=cpu_device,
        rtol=1e-03,
        atol=1e-06,
        num_tests=100,
        input_size=(1, 3, 64, 64)), "Fused model is not equivalent to the original model!"

    # Get quantization configure for post training quantization
    qcf = torch.quantization.get_default_qat_qconfig(args.backends)

    # Set quantization configure
    net.qconfig = qcf

    # Prepare the quantization for all parts
    torch.quantization.prepare_qat(net, inplace=True)


    net.train()
    # Prepare hyper files and DDP mode
    args.total_batch_size = args.batch_size
    device = select_device(args.device, batch_size=args.batch_size)

    if args.local_rank != -1:
        print(torch.cuda.device_count(), args.local_rank)
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        assert not args.image_weights, '--image-weights argument is not compatible with DDP training'
        args.batch_size = args.total_batch_size // args.world_size

    # Hyperparameters
    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    #Change this line to modify training epochs for each experiment
    total_epoches = args.epochs_each_experiment
    best_map = 0
    data_path = "./data/vip_new.yaml"
    for i in range(total_epoches):
        net = train(hyp, args, device, net)
        net = net.to(cpu_device)
        # Convert all parts into quantized ones
        quantized_net = torch.quantization.convert(net, inplace=False)
        quantized_net.eval()


        print("Here test int8 acc---------------------------------------")
        results, _, _ =test(data_path,
            batch_size = args.batch_size,
            imgsz=416,
            model=quantized_net,
            verbose=True,
            half_precision=False,
            opt=args)

        map = results[3]
        # Change this condition to select best mAP to output
        if True:
            best_map = map
            # Use some images from the folder for evaluation or calibration
            for image_name in pbar:
                image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (416, 416))
                image = torch.tensor(image)
                image = image.view(1, 3, 416, 416)
                image = image.to(cpu_device, non_blocking=True).float() / 255.0
                pred = quantized_net(image)
            wt_compare_dict = ns.compare_weights(fp_net.state_dict(), quantized_net.state_dict())
            # Compare each layer of part 1 and 2, the higher score the better
            for key in wt_compare_dict:
                print(key, compute_error(wt_compare_dict[key]['float'], wt_compare_dict[key]['quantized'].dequantize()))

            # Get the backend and save the torchscript files to the specific path
            backend = args.backends
            traced_module = torch.jit.trace(quantized_net, image, strict=True)

            torch.jit.save(traced_module, '{}/yolov5s_{}_{}_{}.torchscript'.format(args.save_dir, str(i+1), str(best_map),backend))

