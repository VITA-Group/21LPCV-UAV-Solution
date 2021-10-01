import json
import os
import cv2
import numpy as np
import re
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

dir_path = os.path.dirname(os.path.realpath(__file__))

from utils.experimental import (load_selected_labels_csv, load_raw_labels_json, init_pd_csv_reader, id_correction)
from utils.datasets import LoadImages
from utils.draw_tool import DrawTool


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
    calibrated_labels_path = os.path.join(dir_path, 'data/outputs', args.video_name, 'calibrated_track_labels')
    calibrated_video_path = os.path.join(dir_path, 'data/outputs', args.video_name, 'calibrated_video')


    if not os.path.exists(calibrated_labels_path):
        os.makedirs(calibrated_labels_path)

    if not os.path.exists(calibrated_video_path):
        os.makedirs(calibrated_video_path)

    video_annot_dct = load_raw_labels_json(raw_labels_path)

    current_file_data = init_pd_csv_reader(gts_path)

    track_gt_color_idMap = {}
    track_gt_noncolor_idMap = {}


    '''
    4p1b's groundtruth annotation does not have corresponding bbox annotation
    '''
    if persons_count == 4 and balls_count == 1:
        track_gt_noncolor_idMap = {2: 1, 1: 2, 4: 3, 3: 4, 101: 5}
    else:
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
                print(frame_idx)
                id_correction(gts_copy, annot_bboxes_copy, track_gt_color_idMap, track_gt_noncolor_idMap)
            frame_idx += 1


    track_gt_color_idMap[0] = 0
    track_gt_noncolor_idMap[0] = 0
    print(track_gt_color_idMap)
    print(track_gt_noncolor_idMap)


    for frame_idx, annots in video_annot_dct.items():
        if np.any(annots[:, 1] > 100):
            for i in range(annots.shape[0]):
                annots[i, 1] = track_gt_noncolor_idMap[annots[i, 1]]
        else:
            for i in range(annots.shape[0]):
                annots[i, 1] = track_gt_color_idMap[annots[i, 1]]

    vid_writer = None

    dataset = LoadImages(dataset_path)

    frame_idx = 0
    for path, img_orig, vid_cap in dataset:
        if vid_writer is None:
            fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(os.path.join(calibrated_video_path, '{}.m4v'.format(args.video_name)),
                                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

        if frame_idx in video_annot_dct:
            print(frame_idx)
            annot_bboxes = video_annot_dct[frame_idx]
            # t,l,w,h -> t,l,b,r
            annot_bboxes[:, 4:6] = annot_bboxes[:, 2:4] + annot_bboxes[:, 4:6]
            clses, ids, bboxes_ltrb = annot_bboxes[:, 0], annot_bboxes[:, 1], annot_bboxes[:, 2:6]
            DrawTool.draw_frame_idx(img_orig, frame_idx, 'GroundTruth')
            DrawTool.draw_tracks(img_orig, bboxes_ltrb, ids, clses)

            vid_writer.write(img_orig)
            with open(os.path.join(calibrated_labels_path, '{:06d}.json'.format(frame_idx)), 'w') as f:
                annot_bboxes_text = []
                for i, record in enumerate(annot_bboxes):
                    record_text = {}
                    if record[0] == 0:
                        record_text['class'] = 'person'
                    else:
                        record_text['class'] = 'ball'
                        record_text['occlusion'] = int(record[6])
                    record_text['id'] = int(record[1])
                    record_text['bounding box (ltrb)'] = record[2:6].tolist()
                    annot_bboxes_text.append(record_text)
                json.dump(annot_bboxes_text, f)
        frame_idx += 1
