import os
import cv2
import numpy as np

import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

dir_path = os.path.dirname(os.path.realpath(__file__))

from utils.experimental import load_labels_from_csv
from utils.datasets import LoadImages
from utils.draw_tool import DrawTool


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='5p2b_01A1', help='source')
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

    dataset_path = os.path.join(dir_path, 'data/inputs', args.video_name)
    gt_init_tracks_path = os.path.join(dir_path, 'data/inputs', args.video_name, '{}_init.csv'.format(args.video_name))

    debugging_video_path = os.path.join(dir_path, 'data/outputs', os.path.basename(args.video_name), 'calibration')

    if not os.path.exists(debugging_video_path):
        os.makedirs(debugging_video_path)

    resolution = video_resolution_dct[args.video_name]
    gt_tracks_dct = load_labels_from_csv(gt_init_tracks_path, img_h=resolution[0], img_w=resolution[1])

    vid_writer = None

    dataset = LoadImages(dataset_path)

    frame_idx = 0
    for path, img_orig, vid_cap in dataset:
        print(frame_idx)
        img_gt_tracks = np.copy(img_orig)

        DrawTool.draw_frame_idx(img_gt_tracks, frame_idx, 'GroundTruth Tracks')

        repeat = 1

        if vid_writer is None:
            fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(os.path.join(debugging_video_path, '{}.m4v'.format(args.video_name)), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))


        if frame_idx in gt_tracks_dct:
            gt_tracks = gt_tracks_dct[frame_idx]
            # t,l,w,h -> t,l,b,r
            gt_tracks[:, 4:6] = gt_tracks[:, 2:4] + gt_tracks[:, 4:6]
            gt_clses, gt_ids, gt_bboxes_ltrb = gt_tracks[:, 0], gt_tracks[:, 1], gt_tracks[:, 2:6]
            DrawTool.draw_tracks(img_gt_tracks, bboxes_ltrb=gt_bboxes_ltrb, ids=gt_ids, clses=gt_clses)
            for _ in range(repeat):
                vid_writer.write(img_gt_tracks)
        frame_idx += 1
