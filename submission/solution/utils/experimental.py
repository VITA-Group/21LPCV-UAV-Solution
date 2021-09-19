import re
import pickle
import os
import json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment


def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def load_raw_labels_json(src):
    print(src)
    matches = re.search(r'(\d)p(\d)b_', src)
    persons_count = int(matches[1])

    color_id_map = {'blue': 11, 'purple': 12, 'red': 13, 'orange': 14, 'yellow': 15, 'green': 16}

    video_annot_dct = {}
    for f in os.listdir(src):
        if not f.endswith('.json'): continue
        frame_idx = int(f.split('.')[0])
        with open(os.path.join(src, f), "r") as read_file:
            data = json.load(read_file)
            # record = (class, id, x, y, w, h, occlusion)
            frame_annot = np.zeros((len(data), 7), dtype=int)
            for i, record in enumerate(data):
                for key, value in record.items():
                    if key == 'label_name':
                        cls_name = value.split('_')[0]
                        if cls_name == 'ball':
                            if len(value.split('_')) == 1:
                                id = 1 + 100
                            elif not value.split('_')[1].isnumeric():
                                id = color_id_map[value.split('_')[1]]
                            else:
                                id = int(value.split('_')[1]) + 100
                            cls = 1
                            frame_annot[i, :2] = cls, id
                        else:
                            cls = 0
                            id = int(value.split('_')[1])
                            if id > persons_count:  # Exclude those annotated persons out of scene
                                break
                            frame_annot[i, :2] = cls, id
                    if key == 'pos':
                        frame_annot[i, 2:6] = value
                    if key == 'occlusion':
                        frame_annot[i, 6] = value
            frame_annot = frame_annot[~np.all(frame_annot == 0, axis = 1)]
            video_annot_dct[frame_idx] = frame_annot

    return video_annot_dct


def load_labels_json(src):
    print(src)
    matches = re.search(r'(\d)p(\d)b_', src)
    persons_count = int(matches[1])

    video_annot_dct = {}
    for f in os.listdir(src):
        if not f.endswith('.json'): continue
        frame_idx = int(f.split('.')[0])
        with open( os.path.join(src, f), "r") as read_file:
            data = json.load(read_file)
            # record = (class, id, x, y, w, h, occlusion)
            frame_annot = np.zeros((len(data), 6), dtype=int)
            # frame_annot = np.zeros((len(data), 7), dtype=int)
            for i, record in enumerate(data):
                for key, value in record.items():
                    if key == 'class':
                        if value == 'ball':
                            cls = 1
                        else:
                            cls = 0
                        frame_annot[i, 0] = cls
                    if key == 'id':
                        frame_annot[i, 1] = value
                    if key == 'bounding box (ltrb)':
                        frame_annot[i, 2:6] = value
                    # if key == 'occlusion':
                    #     frame_annot[i, 6] = value
            frame_annot = frame_annot[~np.all(frame_annot == 0, axis = 1)]
            video_annot_dct[frame_idx] = frame_annot

    return video_annot_dct


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


def load_labels_csv(frame_number, current_file_data):
    '''
    Parameter:
        file_name:      path to the label file. groundtruths.txt
        image_width:    the width of image (video frame)
        image_height:   the height of image (video frame)
        frame_number:   the specific frame number that we want
                        if we want the whole label table the this should be -1
                        the default value is -1
    Return:
        When frame_number is -1:
            type:       pandas DataFrame
            content:    all labels
            format:     ["Frame", "Class","ID","X","Y","Width","Height"]
        When frame_number is not -1:
            type:       pytorch tensor
            content:    coordinates of objects in the requested frame
                        empty tensor if the requested frame doesn't exist in the label file
            format:     ["Class","ID","X","Y","Width","Height"]
    '''
    frame = current_file_data[(current_file_data["Frame"] == frame_number)]
    pt_frame = np.asarray(frame[["Class", "ID", "X", "Y", "Width", "Height"]].values)
    return pt_frame


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


def iou_cost(gts, annots, gts_indices, annots_indices):
    def IoU(bboxes, candidates, epsilon=1e-5):
        # pairwise jaccard botween boxes a and boxes b
        # box: [left, top, right, bottom]
        tl = np.maximum(bboxes[:, np.newaxis, :2], candidates[:, :2])
        br = np.minimum(bboxes[:, np.newaxis, 2:], candidates[:, 2:])
        inter = np.clip(br - tl, a_min=0, a_max=None)

        area_int = np.prod(inter, axis=2)
        area_bboxes = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
        area_candidates = np.prod(candidates[:, 2:] - candidates[:, :2], axis=1)

        area_union = area_bboxes[:, np.newaxis] + area_candidates - area_int
        return area_int / np.clip(area_union, a_min=epsilon, a_max=None)  # shape: (len(a) x len(b))

    bboxes = np.asarray([gts[idx] for idx in gts_indices])
    candidates = np.asarray([annots[idx] for idx in annots_indices])

    cost_matrix = IoU(bboxes, candidates)

    if cost_matrix.ndim == 1:
        cost_matrix = cost_matrix[:, np.newaxis]
    cost_matrix = np.multiply(np.ones_like(cost_matrix) - cost_matrix, 0.5)

    return cost_matrix


def min_cost_matching(distance_metric, max_distance, gts, annots):
    gts_indices = np.arange(gts.shape[0])
    annots_indices = np.arange(annots.shape[0])

    cost_matrix = distance_metric(
        gts, annots, gts_indices, annots_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_gts, unmatched_annots = [], [], []
    for col, annot_idx in enumerate(annots_indices):
        if col not in col_indices:
            unmatched_annots.append(annot_idx)
    for row, gt_idx in enumerate(gts_indices):
        if row not in row_indices:
            unmatched_gts.append(gt_idx)
    for row, col in zip(row_indices, col_indices):
        gt_idx = gts_indices[row]
        annot_idx = annots_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_gts.append(gt_idx)
            unmatched_annots.append(annot_idx)
        else:
            matches.append((gt_idx, annot_idx))
    return matches, unmatched_gts, unmatched_annots

# clses, ids, bboxes_tlbr = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2:6]
def id_correction(gts, annots, track_gt_color_idMap, track_gt_noncolor_idMap):
    matches, unmatched_gts, unmatched_annots = \
        min_cost_matching(iou_cost, 0.125, gts[:, 2:6], annots[:, 2:6])
    if unmatched_gts or unmatched_annots:
        print('unmatched ground-truths annotation or unofficial annotation')
        return
    # assert not unmatched_gts and not unmatched_annots, 'unmatched official ground-truths or our annotation'

    if np.any(annots[:, 1] > 100):
        for gt_indx, annot_idx in matches:
            track_gt_noncolor_idMap[annots[annot_idx][1]] = gts[gt_indx][1]
    else:
        for gt_indx, annot_idx in matches:
            track_gt_color_idMap[annots[annot_idx][1]] = gts[gt_indx][1]
    return