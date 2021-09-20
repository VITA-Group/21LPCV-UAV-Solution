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

from utils.datasets import LoadImages
from utils.draw_tool import draw_frame_idx, draw_tracks, draw_bp_assoc, draw_reid_errors, draw_unmatched_errors
from utils.experimental import save_pkl, load_pkl, load_labels_json, load_labels_from_csv


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, required=True, help='source')
    parser.add_argument('--pause_duration', type=int, default=60, help='the pause duration for catch action')
    return parser

video_resolution_dct = {
    '4p1b_01A2': (2160, 3840),
    '5p2b_01A1': (2160, 3840),
    '5p4b_01A2': (2160, 3840),
    '5p5b_03A1': (2160, 3840),
    '7p3b_02M': (1080, 1920),
}

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    src, dst = os.path.join(dir_path, 'data/inputs', args.video_name), os.path.join(dir_path, 'data/outputs', args.video_name)

    gt_init_tracks_path = os.path.join(src, '{}_init.csv'.format(args.video_name))
    gt_tracks_path = os.path.join(src, 'labels')
    pseudo_tracks_path = os.path.join(src, 'pseudo_labels')
    gt_action_labels_path = os.path.join(src, '{}_action_noisy.csv'.format(args.video_name))

    pred_tracks_path = os.path.join(dst, 'offline_action_detection')
    pred_action_labels_path = os.path.join(dst, '{}_out.csv'.format(args.video_name))

    debugging_video_path = os.path.join(dst, 'debugging')
    if not os.path.exists(debugging_video_path):
        os.makedirs(debugging_video_path)

    resolution = video_resolution_dct[args.video_name]
    gt_init_tracks_dct = load_labels_from_csv(gt_init_tracks_path, img_h=resolution[0], img_w=resolution[1])

    gt_tracks_dct = load_labels_json(gt_tracks_path)

    gt_tracks_dct.update(gt_init_tracks_dct)

    pseudo_tracks_dct = load_labels_json(pseudo_tracks_path)

    gt_tracks_dct = {**gt_tracks_dct, **pseudo_tracks_dct}

    gt_video_ballcolor_dct = {
        '4p1b_01A2' : ['5_blue'],
        '5p2b_01A1' : ['6_red', '7_yellow'],
        '5p4b_01A2' : ['6_yellow', '7_purple', '8_red', '9_green'],
        '5p5b_03A1' : ['6_orange', '7_blue', '8_red', '9_yellow', '10_purple'],
        '7p3b_02M'  : ['12_yellow', '13_red', '14_green']
    }

    gt_csv_columns = ['frame'] + gt_video_ballcolor_dct[args.video_name]
    gt_action_labels = pd.read_csv(gt_action_labels_path)
    df_gt = pd.DataFrame(gt_action_labels, columns=tuple(gt_csv_columns)).to_numpy()
    print(df_gt.shape[0], df_gt.shape[1])


    pred_video_ballcolor_dct = {
        '4p1b_01A2': ['5'],
        '5p2b_01A1': ['6', '7'],
        '5p4b_01A2': ['6', '7', '8', '9'],
        '5p5b_03A1': ['6', '7', '8', '9', '10'],
        '7p3b_02M': ['12', '13', '14']
    }

    pred_csv_columns = ['frame'] + pred_video_ballcolor_dct[args.video_name]
    pred_action_labels = pd.read_csv(pred_action_labels_path)
    df_pred = pd.DataFrame(pred_action_labels, columns=tuple(pred_csv_columns)).to_numpy()
    print(df_pred.shape[0], df_pred.shape[1])

    tracks_data = load_pkl(os.path.join(pred_tracks_path, 'tracks_history.pkl'))
    frames_idx_data = load_pkl(os.path.join(pred_tracks_path, 'frames_idx_history.pkl'))
    unmatched_detections_history = load_pkl(os.path.join(pred_tracks_path, 'unmatched_detections_history.pkl'))
    unmatched_tracks_history = load_pkl(os.path.join(pred_tracks_path, 'unmatched_tracks_history.pkl'))

    pred_tracks_dct = {}
    for i, frame_idx in enumerate(frames_idx_data):
        pred_tracks_dct[frame_idx] = tracks_data[i][:, (5, 4, 0, 1, 2, 3)]

    vid_writer = None

    dataset = LoadImages(src)

    frame_idx = 0
    gt_action_indices = df_gt[:, 0].tolist()
    pred_action_indices = df_pred[:, 0].tolist()
    prev_assocs_pred = [(12,0), (13,0), (14,0)]
    prev_assocs_gt = [(12,0), (13,0), (14,0)]
    for path, img_orig, vid_cap in dataset:
        if frame_idx == 3700:
            print(frame_idx)
        img_pred_tracks = np.copy(img_orig)
        img_gt_tracks = np.copy(img_orig)
        img_reid_error = np.copy(img_orig)
        img_unmatched_error = np.copy(img_orig)

        draw_frame_idx(img_pred_tracks, frame_idx, 'Predicted Tracks')
        draw_frame_idx(img_reid_error, frame_idx, 'ReID Error')
        draw_frame_idx(img_unmatched_error, frame_idx, 'Unmatched Error')
        draw_frame_idx(img_gt_tracks, frame_idx, 'GroundTruth Tracks')

        repeat = 1

        if vid_writer is None:
            fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(
                vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2
            vid_writer = cv2.VideoWriter(os.path.join(debugging_video_path, '{}.m4v'.format(args.video_name)), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))


        if frame_idx in gt_tracks_dct:
            gt_tracks = gt_tracks_dct[frame_idx]
            gt_clses, gt_ids, gt_bboxes_ltrb = gt_tracks[:, 0], gt_tracks[:, 1], gt_tracks[:, 2:6]
            draw_tracks(img_gt_tracks, bboxes_ltrb=gt_bboxes_ltrb, ids=gt_ids, clses=gt_clses)
            if frame_idx in gt_action_indices:
                current_assocs_gt = list(zip(pred_video_ballcolor_dct[args.video_name],
                                             df_gt[gt_action_indices.index(frame_idx), 1:].tolist()))
                draw_bp_assoc(img_gt_tracks, gt_tracks, prev_assocs_gt, current_assocs_gt)
                prev_assocs_gt = current_assocs_gt
                repeat = args.pause_duration

        if frame_idx in pred_tracks_dct:
            pred_tracks = pred_tracks_dct[frame_idx]
            pred_clses, pred_ids, pred_bboxes_ltrb = pred_tracks[:, 0], pred_tracks[:, 1], pred_tracks[:, 2:6]
            draw_tracks(img_pred_tracks, bboxes_ltrb=pred_bboxes_ltrb, ids=pred_ids, clses=pred_clses)
            if frame_idx in pred_action_indices:
                current_assocs_pred = list(zip(pred_video_ballcolor_dct[args.video_name],
                                               df_pred[pred_action_indices.index(frame_idx), 1:].tolist()))
                draw_bp_assoc(img_pred_tracks, pred_tracks, prev_assocs_pred, current_assocs_pred)
                prev_assocs_pred = current_assocs_pred
                repeat = args.pause_duration


        if frame_idx in pred_tracks_dct and frame_idx in gt_tracks_dct:
            pred_tracks = pred_tracks_dct[frame_idx]
            gt_tracks = gt_tracks_dct[frame_idx]
            has_reid_error = draw_reid_errors(img_reid_error, pred_tracks, gt_tracks)
            repeat = 10 if repeat == 1 and has_reid_error else repeat


        if frame_idx in pred_tracks_dct:
            unmatched_detects, unmatched_tracks = [], []
            has_unmatched_error = False
            if frame_idx in unmatched_detections_history:
                unmatched_detects = unmatched_detections_history[frame_idx]
            if frame_idx in unmatched_tracks_history:
                unmatched_tracks = unmatched_tracks_history[frame_idx]
            if frame_idx in unmatched_detections_history or frame_idx in unmatched_tracks_history:
                draw_unmatched_errors(img_unmatched_error, unmatched_tracks, unmatched_detects)
                repeat = 10 if repeat == 1 else repeat

        for _ in range(repeat):
            outcome = np.concatenate(
                            (np.concatenate((img_pred_tracks, img_gt_tracks), axis=0),
                             np.concatenate((img_reid_error, img_unmatched_error), axis=0)
                             ), axis=1)
            vid_writer.write(outcome)
        frame_idx += 1
        # if frame_idx >= 150:
        #     break