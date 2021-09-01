import json

import os
from pathlib import Path

import glob
import cv2
import numpy as np
import re
import pandas as pd
import argparse

from debugging_utils import (LoadImages, draw_frame_idx, draw_tracks, load_labels_json)


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, required=True, help='source')
    parser.add_argument('--pause_duration', type=int, default=30, help='the pause duration for catch action')
    return parser

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    matches = re.search(r'(\d)p(\d)b_', args.video_name)

    persons_count = int(matches[1])
    balls_count = int(matches[2])

    dir_path = os.path.dirname(os.path.realpath(__file__))


    dataset_path = os.path.join(dir_path, 'inputs', args.video_name)
    labels_path = os.path.join(dir_path, 'inputs', args.video_name, 'labels')
    gt_action_labels_path = os.path.join(dir_path, 'inputs', args.video_name, '{}_action.csv'.format(args.video_name))
    noisy_action_labels_path = os.path.join(dir_path, 'inputs', args.video_name, '{}_action_noisy.csv'.format(args.video_name))

    corrected_video_path = os.path.join(dir_path, 'outputs', args.video_name, 'correcting_action')
    if not os.path.exists(corrected_video_path):
        os.makedirs(corrected_video_path)


    video_annot_dct = load_labels_json(labels_path)

    video_ballcolor_dct = {
        '4p1b_01A2' : ['5_blue'],
        '5p2b_01A1' : ['6_red', '7_yellow'],
        '5p4b_01A2' : ['6_yellow', '7_purple', '8_red', '9_green'],
        '5p5b_03A1' : ['6_orange', '7_blue', '8_red', '9_yellow', '10_purple'],
        '7p3b_02M'  : ['12_yellow', '13_red', '14_green']
    }

    csv_columns = ['frame'] + video_ballcolor_dct[args.video_name]

    noisy_action_labels = pd.read_csv(noisy_action_labels_path)
    df_noisy = pd.DataFrame(noisy_action_labels, columns=tuple(csv_columns)).to_numpy()
    print(df_noisy.shape[0], df_noisy.shape[1])

    gt_action_labels = pd.read_csv(gt_action_labels_path)
    df_gt = pd.DataFrame(gt_action_labels, columns=tuple(csv_columns)).to_numpy()
    print(df_gt.shape[0], df_gt.shape[1])

    vid_writer = None

    dataset = LoadImages(dataset_path)

    frame_idx = 0
    noisy_action_indices = df_noisy[:, 0].tolist()
    gt_action_indices = df_gt[:, 0].tolist()
    for path, img_orig, vid_cap in dataset:
        img_gt = np.copy(img_orig)
        img_noisy = np.copy(img_orig)
        draw_frame_idx(img_gt, frame_idx, 'GroundTruth')
        draw_frame_idx(img_noisy, frame_idx, 'Noisy')
        repeat = 1
        if vid_writer is None:
            fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2
            vid_writer = cv2.VideoWriter(os.path.join(corrected_video_path, '{}.m4v'.format(args.video_name)),
                                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

        if frame_idx in video_annot_dct and frame_idx in noisy_action_indices:
            annot_bboxes = video_annot_dct[frame_idx]
            draw_tracks(img_noisy, annot_bboxes)
            repeat = args.pause_duration
        if frame_idx in video_annot_dct and frame_idx in gt_action_indices:
            annot_bboxes = video_annot_dct[frame_idx]
            draw_tracks(img_gt, annot_bboxes)
            repeat = args.pause_duration
        for _ in range(repeat):
            vid_writer.write(np.concatenate((img_noisy, img_gt), axis=0))
        frame_idx += 1