import os
import numpy as np
import pandas as pd
import os
import re
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

dir_path = os.path.dirname(os.path.realpath(__file__))

from utils.experimental import (load_selected_labels_csv, load_raw_labels_json, init_pd_csv_reader, id_correction)
from utils.datasets import LoadImages

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, required=True, help='source')
    return parser

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    matches = re.search(r'(\d)p(\d)b_', args.video_name)

    persons_count = int(matches[1])
    balls_count = int(matches[2])

    dataset_path = os.path.join(dir_path, 'data/inputs', args.video_name)
    raw_labels_path = os.path.join(dir_path, 'data/inputs', args.video_name, 'raw_labels')
    gts_path = os.path.join(dir_path, 'data/inputs', args.video_name, '{}_init.csv'.format(args.video_name))
    calibrated_action_labels_path = os.path.join(dir_path, 'data/outputs', args.video_name, 'calibrated_action_labels')


    if not os.path.exists(calibrated_action_labels_path):
        os.makedirs(calibrated_action_labels_path)

    video_annot_dct = load_raw_labels_json(raw_labels_path)

    current_file_data = init_pd_csv_reader(gts_path)

    track_gt_color_idMap = {}
    track_gt_noncolor_idMap = {}

    '''
    5p4b's action annotation id is different from its bbox annotation id
    '''
    if persons_count == 5 and balls_count == 4:
        track_gt_noncolor_idMap = {5: 1, 1: 2, 2: 3, 3: 4, 4: 5, 11: 6, 13: 7, 12: 8, 14: 9, 0: 0}
    elif persons_count == 4 and balls_count == 1:
        track_gt_noncolor_idMap = {2: 1, 1: 2, 4: 3, 3: 4, 5: 5}
    else:
        track_gt_idMap = {}
        dataset = LoadImages(dataset_path)
        frame_idx = 0
        for path, img_orig, vid_cap in dataset:
            img_h, img_w, _ = img_orig.shape  # get image shape
            # cx,cy,w,h
            gts = load_selected_labels_csv(frame_idx, current_file_data)
            gts = np.multiply(gts, np.asarray([1.0, 1.0, img_w, img_h, img_w, img_h])).astype(int)

            if gts.shape[0] != 0 and frame_idx in video_annot_dct:
                annot_bboxes_copy = np.copy(video_annot_dct[frame_idx])
                annot_bboxes_copy[np.arange(annot_bboxes_copy.shape[0])] = annot_bboxes_copy[np.argsort(annot_bboxes_copy[:, 1])]
                gts_copy = np.copy(gts)
                gts_copy[np.arange(gts_copy.shape[0])] = gts_copy[np.argsort(gts_copy[:, 1])]

                # cx,cy,w,h -> t,l,w,h
                gts_copy[:, 2:4] = gts_copy[:, 2:4] - 0.5 * gts_copy[:, 4:6]

                # t,l,w,h -> t,l,b,r
                gts_copy[:, 4:6] = gts_copy[:, 2:4] + gts_copy[:, 4:6]
                annot_bboxes_copy[:, 4:6] = annot_bboxes_copy[:, 2:4] + annot_bboxes_copy[:, 4:6]

                id_correction(gts_copy, annot_bboxes_copy, track_gt_color_idMap, track_gt_noncolor_idMap)
            frame_idx += 1

    track_gt_color_idMap[0] = 0
    track_gt_noncolor_idMap[0] = 0
    print(track_gt_color_idMap)
    print(track_gt_noncolor_idMap)


    if len(track_gt_color_idMap) > 1:
        track_gt_idMap = track_gt_color_idMap
    else:
        track_gt_idMap = track_gt_noncolor_idMap


    video_ballcolor_dct = {
        '4p1b_01A2' : ['5_blue'],
        '5p2b_01A1' : ['6_red', '7_yellow'],
        '5p4b_01A2' : ['6_yellow', '7_purple', '8_red', '9_green'],
        '5p5b_03A1' : ['6_orange', '7_blue', '8_red', '9_yellow', '10_purple'],
        '7p3b_02M'  : ['12_yellow', '13_red', '14_green']
    }

    data = pd.read_excel(os.path.join('data/inputs', args.video_name, 'raw_action_labels', '{}.xlsx'.format(args.video_name)))
    columns = ['frame'] + video_ballcolor_dct[args.video_name]
    df = pd.DataFrame(data, columns= tuple(columns))

    calibrated_labels = []
    for index, row in df.iterrows():
        calibrated_labels.append([row['frame']] + [track_gt_idMap[row[col_name]] for col_name in video_ballcolor_dct[args.video_name]])
    print(df.shape[0], df.shape[1])
    print(df.to_numpy().shape)

    df_calibrated = pd.DataFrame(calibrated_labels, columns=tuple(columns))
    df_calibrated.to_csv(os.path.join(calibrated_action_labels_path, '{}_action_labels.csv'.format(args.video_name)), index=False)
    print(df_calibrated.shape[0], df.shape[1])
    print(df_calibrated.to_numpy().shape)
