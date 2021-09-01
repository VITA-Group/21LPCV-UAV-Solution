import json

import os
from pathlib import Path

import glob
import cv2
import numpy as np
import re
import pandas as pd

import pickle

import argparse
import sys
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from debugging_utils import (LoadImages, draw_frame_idx, draw_tracks)


def init_pd_csv_reader(file_name):
    if not os.path.exists(file_name):
        print("The file", file_name, "doesn't exist.")
        exit(1)
    current_file_data = pd.read_csv(file_name, sep=',')
    return current_file_data


def load_labels(current_file_data, frame_number=-1):
    frame = current_file_data[(current_file_data["Frame"] == frame_number)]
    gt_labels = torch.tensor(frame[["Class", "ID", "X", "Y", "Width", "Height"]].values)
    gt_labels = torch.mul(gt_labels, torch.tensor([1.0, 1.0, 1920*2, 1080*2, 1920*2, 1080*2]))
    return gt_labels


def load_labels_from_csv(filename):
    gt_annots_dct = {}
    csv_file_data = init_pd_csv_reader(filename)
    unique_frames = np.unique(csv_file_data["Frame"].to_numpy())

    for frame in unique_frames:
        labels = load_labels(csv_file_data, frame_number=frame).detach().numpy()
        labels[:, 2:4] -= 0.5 * labels[:, 4:6]
        labels = labels.astype(int)
        gt_annots_dct[frame] = labels
    return gt_annots_dct

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='5p2b_01A1', help='source')
    parser.add_argument('--pause_duration', type=int, default=60, help='the pause duration for catch action')
    return parser

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    dataset_path = os.path.join(dir_path, 'inputs', args.video_name)
    gt_labels_path = os.path.join(dir_path, 'inputs', args.video_name, '{}_init.csv'.format(args.video_name))



    debugging_video_path = os.path.join(dir_path, 'outputs', os.path.basename(args.video_name), 'debugging')

    if not os.path.exists(debugging_video_path):
        os.makedirs(debugging_video_path)

    gt_tracks_dct = load_labels_from_csv(gt_labels_path)

    vid_writer = None

    dataset = LoadImages(dataset_path)

    frame_idx = 0
    for path, img_orig, vid_cap in dataset:
        print(frame_idx)
        img_gt_tracks = np.copy(img_orig)

        draw_frame_idx(img_gt_tracks, frame_idx, 'GroundTruth Tracks')

        repeat = 1

        if vid_writer is None:
            fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(os.path.join(debugging_video_path, '{}.m4v'.format(args.video_name)), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))


        if frame_idx in gt_tracks_dct:
            gt_tracks = gt_tracks_dct[frame_idx]
            gt_clses, gt_ids, gt_bboxes_ltwh = gt_tracks[:, 0], gt_tracks[:, 1], gt_tracks[:, 2:6]
            draw_tracks(img_gt_tracks, bboxes_ltwh=gt_bboxes_ltwh, ids=gt_ids, clses=gt_clses)
            for _ in range(repeat):
                vid_writer.write(img_gt_tracks)
        frame_idx += 1
