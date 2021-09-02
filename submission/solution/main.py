import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import numpy as np
# https://github.com/pytorch/pytorch/issues/3678
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils.datasets import LoadImages, letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, postprocess)
from utils.experimental import (save_pkl, load_pkl)

from parser import get_config
from deep_sort import DeepSort


import pandas as pd

from online_action_detector import OnlineActionDetector
from offline_action_detector import OfflineActionDetector

from enums import ObjectCategory
from detection import Detection
from feature_extractor import Extractor
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

torch.backends.quantized.engine = 'qnnpack'

class CropActivityRegion(object):
    def __init__(self, extra):
        self.latest_det = torch.empty(0, 4).to('cpu')
        self.extra = extra

    '''
    Update the cropper with the latest detection results
    '''
    def update_memory(self, det):
        self.latest_det = np.copy(det)

    '''
    Coordinates Transformation
    '''
    def coords_final2orig(self, det, img_crop_letterbox, img_crop, dx, dy):
        det_orig = np.copy(det)
        det_orig[:, :4] = scale_coords(img_crop_letterbox.shape[2:], torch.tensor(det_orig[:, :4]), img_crop.shape).round()
        det_orig[:, [0, 2]] += dx
        det_orig[:, [1, 3]] += dy
        return det_orig

    def crop_image(self, img, bbox):
        '''
        Crop the image based on the location of bboxs and add extra space to
        make sure the cropped activity region could cover all persons and balls.
        '''
        # 1. expand the bbox
        h, w = img.shape[:2]
        # print('debug:', H, W)
        xmin = w
        ymin = h
        xmax = ymax = 0
        for x1, y1, x2, y2 in bbox[:, :4]:
            xmin = int(min(xmin, x1 * (1 - self.extra)))
            ymin = int(min(ymin, y1 * (1 - self.extra)))
            xmax = int(min(w, max(xmax, x2 * (1 + self.extra))))
            ymax = int(min(h, max(ymax, y2 * (1 + self.extra))))

        return img[int(ymin): int(ymax), int(xmin): int(xmax), :], xmin, ymin

    def letterbox_image(self, img_crop, imgsz, stride):
        img_crop_letterbox, _, _ = letterbox(img_crop, new_shape=imgsz, stride=stride)
        img_crop_letterbox = img_crop_letterbox[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_crop_letterbox = np.ascontiguousarray(img_crop_letterbox)

        img_crop_letterbox = torch.from_numpy(img_crop_letterbox).to('cpu')
        img_crop_letterbox = img_crop_letterbox.float()  # uint8 to fp16/32
        img_crop_letterbox /= 255.0  # 0 - 255 to 0.0 - 1.0
        img_crop_letterbox = img_crop_letterbox.unsqueeze(
            0) if img_crop_letterbox.ndimension() == 3 else img_crop_letterbox
        return img_crop_letterbox

    def set_extra_ratio(self, extra):
        self.extra = extra


class CropsBoxesCache(object):
    def __init__(self, key_frames, gt_frames, persons_count, balls_count):
        self.key_frames = key_frames
        self.gt_frames = gt_frames

        self.gt_persons_crop_cache = np.zeros((len(gt_frames), persons_count, 128, 64, 3), dtype=np.uint8)
        self.gt_balls_crop_cache = np.zeros((len(gt_frames), balls_count, 128, 64, 3), dtype=np.uint8)

        self.gt_persons_box_cache = np.zeros((len(gt_frames), persons_count, 4), dtype=np.int32)
        self.gt_balls_box_cache = np.zeros((len(gt_frames), balls_count, 4), dtype=np.int32)


        self.key_persons_crop_cache = np.zeros((len(key_frames), persons_count, 128, 64, 3), dtype=np.uint8)
        self.key_balls_crop_cache = np.zeros((len(key_frames), balls_count, 128, 64, 3), dtype=np.uint8)

        self.key_persons_box_cache = np.zeros((len(key_frames), persons_count, 4), dtype=np.int32)
        self.key_balls_box_cache = np.zeros((len(key_frames), balls_count, 4), dtype=np.int32)


    def update(self, frame_idx, crops, boxes, is_person):
        num_detections = crops.shape[0]
        if frame_idx in self.key_frames:
            pos = self.key_frames.index(frame_idx)
            if is_person:
                self.key_persons_crop_cache[pos][:num_detections]= crops
                self.key_persons_box_cache[pos][:num_detections] = boxes
            else:
                self.key_balls_crop_cache[pos][:num_detections] = crops
                self.key_balls_box_cache[pos][:num_detections] = boxes
        if frame_idx in self.gt_frames:
            pos = self.gt_frames.index(frame_idx)
            if is_person:
                self.gt_persons_crop_cache[pos][:num_detections] = crops
                self.gt_persons_box_cache[pos][:num_detections] = boxes
            else:
                self.gt_balls_crop_cache[pos][:num_detections] = crops
                self.gt_balls_box_cache[pos][:num_detections] = boxes

    def fetch(self, frame_idx, is_person):
        if frame_idx in self.key_frames:
            pos = self.key_frames.index(frame_idx)
            if is_person:
                crops = self.key_persons_crop_cache[pos]
                boxes = self.key_persons_box_cache[pos]
            else:
                crops = self.key_balls_crop_cache[pos]
                boxes = self.key_balls_box_cache[pos]
        if frame_idx in self.gt_frames:
            pos = self.gt_frames.index(frame_idx)
            if is_person:
                crops = self.gt_persons_crop_cache[pos]
                boxes = self.gt_persons_box_cache[pos]
            else:
                crops = self.gt_balls_crop_cache[pos]
                boxes = self.gt_balls_box_cache[pos]
        return crops, boxes


class Solution(object):
    def __init__(self, opt):
        self.opt = opt
        self.gt_frames, self.gt_labels_history, self.gt_pids, self.gt_bids = self.read_csv_gt_tracks(opt.groundtruths)


        self.frame_sample_rate = opt.skip_frames  # Sample rate used to pick the frame at a fixed temporal stride
        self.current_file_name = opt.groundtruths  # The groundtruth file used to give the initialization and calibration of the tracks
        self.current_file_data = self.init_pd_csv_reader(opt.groundtruths)


        self.track_gt_idMap = {}  # The mapping between track ID and groundtruth ID
        self.vid_path, self.vid_writer = None, None  # The video that visualizes tracking results

        # make new output folder
        if not os.path.exists(opt.output):
            os.makedirs(opt.output)

        '''
        Initialize deepsort online tracker
        '''
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        self.extractor = Extractor(os.path.join(dir_path, cfg.DEEPSORT.REID_CKPT))

        self.deepsort = DeepSort(max_dist=cfg.DEEPSORT.MAX_DIST, nn_budget=cfg.DEEPSORT.NN_BUDGET)

        '''
        Initialize person/ball detector
        '''
        self.grid = torch.load(os.path.join(dir_path, 'weights/grid.pt'), map_location='cpu')
        self.yolo = torch.jit.load(opt.yolo_weights, map_location='cpu')
        self.online_action_detector = OnlineActionDetector(ball_ids=self.gt_bids,
                                                           person_ids=self.gt_pids)

        '''
        Initialize Dataloader
        '''
        model_stride = torch.tensor([8, 16, 32])
        self.stride = int(model_stride.max())  # model stride
        self.imgsz = check_img_size(opt.img_size, s=self.stride)  # check img_size w.r.t. model stride
        # print(self.imgsz)
        self.dataset = LoadImages(opt.source, img_size=self.imgsz, stride=self.stride)

        '''
        Initialize activity region cropper
        '''
        self.crop_activity_region = CropActivityRegion(extra=0.25)

        self.tracks_history = []
        self.frames_idx_history = []
        self.unmatched_tracks_history = {}
        self.unmatched_detections_history = {}

        key_frames = np.arange(self.dataset.nframes).tolist()
        self.key_frames = [idx for idx in key_frames if idx not in self.gt_frames and idx % self.frame_sample_rate == 0]
        self.cache = CropsBoxesCache(self.key_frames, self.gt_frames, len(self.gt_pids), len(self.gt_bids))

    def save_frames_tracks_history(self, tracks, frame_idx):
        # tracks_cxcywh[:, :2] = 0.5 * (tracks[:, :2] + tracks[:, 2:4])
        tracks_ltwh = np.zeros_like(tracks[:, :6], dtype=int)
        tracks_ltwh[:, :2] = tracks[:, :2]
        tracks_ltwh[:, 2:4] = tracks[:, 2:4] - tracks[:, :2]
        tracks_ltwh[:, 4:6] = tracks[:, 4:6]

        self.tracks_history.append(tracks_ltwh)
        self.frames_idx_history.append(frame_idx)

    def collision(self, ball_dets, person_dets):
        balls_center = 0.5 * (ball_dets[:, :2] + ball_dets[:, 2:4])
        persons_lt, persons_rb = person_dets[:, :2], person_dets[:, 2:4]
        # persons_lt = person_dets[:, :2] - 0.5 * person_dets[:, 2:4]
        # persons_rb = person_dets[:, :2] + 0.5 * person_dets[:, 2:4]
        bp_collistion_matx = np.logical_and(
            np.logical_and(*np.dsplit(
                np.subtract(balls_center[:, np.newaxis, :], persons_lt[np.newaxis, :, :]) > 0, 2)),
            np.logical_and(*np.dsplit(
                np.subtract(persons_rb[np.newaxis, :, :], balls_center[:, np.newaxis, :]) > 0, 2))
        )
        return bp_collistion_matx

    def init_pd_csv_reader(self, file_name):
        if not os.path.exists(file_name):
            print("The file", file_name, "doesn't exist.")
            exit(1)
        current_file_data = pd.read_csv(file_name, sep=',')
        return current_file_data

    def load_labels(self, frame_number=-1):
        frame = self.current_file_data[(self.current_file_data["Frame"] == frame_number)]
        pt_frame = torch.tensor(frame[["Class", "ID", "X", "Y", "Width", "Height"]].values)
        return pt_frame

    def read_csv_gt_tracks(self, filename):
        def load_labels(csv_file_data, frame_number=-1):
            frame = csv_file_data[(csv_file_data["Frame"] == frame_number)]
            return frame[["Class", "ID", "X", "Y", "Width", "Height"]].values

        csv_file_data = self.init_pd_csv_reader(filename)
        unique_frames = np.sort(np.unique(csv_file_data["Frame"].to_numpy())).tolist()
        unique_ids = np.sort(np.unique(csv_file_data["ID"].to_numpy()))

        gt_labels = np.zeros((len(unique_frames), unique_ids.size, 6), dtype=np.float32)

        gt_pids, gt_bids = [], []
        for i, frame_idx in enumerate(unique_frames):
            labels = load_labels(csv_file_data, frame_number=frame_idx)
            gt_labels[i, :labels.shape[0], :] = labels
            gt_bids += labels[labels[:,0] == 1, 1].astype(int).tolist()
            gt_pids += labels[labels[:,0] == 0, 1].astype(int).tolist()
        return unique_frames, gt_labels, np.unique(gt_pids).tolist(), np.unique(gt_bids).tolist()

    def cxcywh2xyxy(self, bboxes_cxcywh, img):
        img_h, img_w, _ = img.shape  # get image shape
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        bboxes_ltbr = bboxes_cxcywh.clone() if isinstance(bboxes_cxcywh, torch.Tensor) else np.copy(bboxes_cxcywh)
        bboxes_ltbr[:, 0:2] = bboxes_cxcywh[:, 0:2] - bboxes_cxcywh[:, 2:4] / 2  # top left
        bboxes_ltbr[:, 2:4] = bboxes_cxcywh[:, 0:2] + bboxes_cxcywh[:, 2:4] / 2  # bottom right
        bboxes_ltbr[:, 0] = np.maximum(np.minimum(bboxes_ltbr[:, 0], img_w - 1), 0)
        bboxes_ltbr[:, 1] = np.maximum(np.minimum(bboxes_ltbr[:, 1], img_h - 1), 0)
        bboxes_ltbr[:, 2] = np.maximum(np.minimum(bboxes_ltbr[:, 2], img_w - 1), 0)
        bboxes_ltbr[:, 3] = np.maximum(np.minimum(bboxes_ltbr[:, 3], img_h - 1), 0)
        return bboxes_ltbr

    def update_gt_cache(self, gts, img, gt_ids, order, frame_idx, is_person):
        crops = self.get_crops(gts, img)
        ordered_crops = np.zeros((len(gt_ids), 128, 64, 3), dtype=np.uint8)
        ordered_boxes = np.zeros((len(gt_ids), 4), dtype=np.int32)
        for i, idx in enumerate(order):
            ordered_crops[idx] = crops[i]
            ordered_boxes[idx] = gts[i, :4]
        self.cache.update(frame_idx, ordered_crops, ordered_boxes, is_person=is_person)

    def update_key_cache(self, dets, img, frame_idx, max_num, is_person):
        crops = self.get_crops(dets, img)
        boxes = dets[:, :4]
        scores = dets[:, 4]
        if scores.size > max_num:
            indices = np.flip(np.argsort(scores)[-max_num:])
        else:
            indices = np.flip(np.argsort(scores))
        sorted_crops = [crops[idx] for idx in indices]
        sorted_crops = np.stack(sorted_crops, axis=0)
        sorted_boxes = [boxes[idx] for idx in indices]
        sorted_boxes = np.stack(sorted_boxes, axis=0)
        self.cache.update(frame_idx, sorted_crops, sorted_boxes, is_person=is_person)

    def run(self):
        outpath = os.path.basename(self.opt.source)[:-4]
        outpath = os.path.join(self.opt.output, outpath + '_out.csv')

        frame_idx = 0

        for path, img_orig, vid_cap in self.dataset:
            img_h, img_w, _ = img_orig.shape  # get image shape
            if frame_idx in self.gt_frames:
                # img_dct[frame_idx] = img_orig
                gts_raw = self.gt_labels_history[self.gt_frames.index(frame_idx)]
                # Remove empty annotation with zeros as placeholder
                gts_raw = gts_raw[~np.all(gts_raw == 0, axis=1)]
                gts_scaled = np.multiply(gts_raw, np.array([1.0, 1.0, img_w, img_h, img_w, img_h]))
                gts_ltrb = self.cxcywh2xyxy(gts_scaled[:, 2:], img_orig)
                # ltrb, id, cls
                gts = np.concatenate((gts_ltrb, gts_scaled[:, 1:2], gts_scaled[:, 0:1]), 1).astype(int)
                gts_person, gts_ball = np.copy(gts[gts[:, 5] == 0]), np.copy(gts[gts[:, 5] == 1])

                self.crop_activity_region.update_memory(gts[:, :4])

                gts_person_ids = gts_person[:, 4]
                persons_order = [self.gt_pids.index(pid) for pid in gts_person_ids]
                gts_ball_ids = gts_ball[:, 4]
                balls_order = [self.gt_bids.index(bid) for bid in gts_ball_ids]

                if gts_person.size > 0:
                    self.update_gt_cache(gts_person, img_orig, self.gt_pids, persons_order, frame_idx, is_person=True)
                if gts_ball.size > 0:
                    self.update_gt_cache(gts_ball, img_orig, self.gt_bids, balls_order, frame_idx, is_person=False)
            else:

                # Sampling at a fixed rate
                if frame_idx % self.frame_sample_rate != 0:
                    frame_idx += 1
                    continue
                # img_dct[frame_idx] = img_orig

                '''
                Crop the activity region from the latest prediction
                (x_min, y_min) and (x_max, y_max) are the coords of the minimum bounding rectangle
                '''
                latest_det = self.crop_activity_region.latest_det
                img_crop, dx, dy = self.crop_activity_region.crop_image(img_orig, latest_det)
                img_crop_letterbox = self.crop_activity_region.letterbox_image(img_crop, imgsz=self.imgsz, stride=self.stride)

                preds = self.yolo(img_crop_letterbox)
                preds = postprocess(preds, self.grid, self.yolo)

                # Apply NMS
                # ltrb, conf, cls
                dets = non_max_suppression(preds[0], self.opt.conf_thres, self.opt.iou_thres, classes=None, agnostic=False)[0]

                if dets.numel() == 0:
                    frame_idx += 1
                    continue

                dets = dets.detach().numpy()

                dets_person = np.copy(dets[dets[:, 5] == 0])
                dets_ball = np.copy(dets[dets[:, 5] == 1])
                if dets_person.shape[0] != 0 and dets_ball.shape[0] != 0:
                    max_ball_height = (dets_person[:, 3] - dets_person[:, 1]).mean(axis=0) * 0.6
                    dets_ball = dets_ball[(dets_ball[:, 3] - dets_ball[:, 1]) < max_ball_height]
                    dets = np.concatenate((dets_person, dets_ball), axis=0)

                dets = self.crop_activity_region.coords_final2orig(dets, img_crop_letterbox, img_crop, dx, dy)
                self.crop_activity_region.update_memory(dets[:, :4])

                # if len(clses_person) < gts_persons.shape[0]:
                self.crop_activity_region.set_extra_ratio(0.1)
                # else:
                #     self.crop_activity_region.set_extra_ratio(0.2)

                dets_person, dets_ball = dets[dets[:,5]==0,:], dets[dets[:,5]==1,:]

                if dets_ball.size == 0 or dets_person.size == 0:
                    self.key_frames.remove(frame_idx)
                    frame_idx += 1
                    continue

                bp_collistion_matx = self.collision(dets_ball, dets_person).reshape(
                                                                    (dets_ball.shape[0], dets_person.shape[0]))
                if not np.any(bp_collistion_matx):
                    self.key_frames.remove(frame_idx)
                    # print(frame_idx)
                    frame_idx += 1
                    continue

                if dets_person.size > 0:
                    self.update_key_cache(dets_person, img_orig, frame_idx, len(self.gt_pids), is_person=True)
                if dets_ball.size > 0:
                    self.update_key_cache(dets_ball, img_orig, frame_idx, len(self.gt_bids), is_person=False)

            frame_idx += 1

        gt_btracks_dct, gt_ptracks_dct = {}, {}
        gt_bdets_dct, gt_pdets_dct = {}, {}
        for frame_idx in range(self.dataset.nframes):
            if frame_idx in self.gt_frames:
                gt_crops_person, gt_boxes_person = self.cache.fetch(frame_idx, is_person=True)
                valid_idx = ~np.all(gt_boxes_person == 0, axis=1)
                gt_boxes_person = gt_boxes_person[valid_idx, ...]
                gt_crops_person = gt_crops_person[valid_idx, ...]
                selected_pids = np.array(self.gt_pids)[valid_idx]

                gt_crops_ball, gt_boxes_ball = self.cache.fetch(frame_idx, is_person=False)
                valid_idx = ~np.all(gt_boxes_ball == 0, axis=1)
                gt_boxes_ball = gt_boxes_ball[valid_idx, ...]
                gt_crops_ball = gt_crops_ball[valid_idx, ...]
                selected_bids = np.array(self.gt_bids)[valid_idx]

                person_detections = self.wrapup_detections(boxes=gt_boxes_person, crops=gt_crops_person, is_person=True)
                ball_detections = self.wrapup_detections(boxes=gt_boxes_ball, crops=gt_crops_ball, is_person=False)

                gt_pdets_dct[frame_idx] = person_detections
                gt_bdets_dct[frame_idx] = ball_detections

                for i, pid in enumerate(selected_pids):
                    if pid in gt_ptracks_dct:
                        gt_ptracks_dct[pid].append(person_detections[i])
                    else:
                        gt_ptracks_dct[pid] = []

                for i, bid in enumerate(selected_bids):
                    if bid in gt_btracks_dct:
                        gt_btracks_dct[bid].append(ball_detections[i])
                    else:
                        gt_btracks_dct[bid] = []

        self.deepsort.tracker_person.initiate_tracks(gt_ptracks_dct, 0)
        self.deepsort.tracker_ball.initiate_tracks(gt_btracks_dct, 1)


        for frame_idx in range(self.dataset.nframes):
            if frame_idx in self.gt_frames:
                # print(frame_idx)

                person_detections, ball_detections = gt_pdets_dct[frame_idx], gt_bdets_dct[frame_idx]
                gt_tracks = self.deepsort.update(person_detections, ball_detections, img_orig)

                if len(gt_tracks) > 0:
                    if self.opt.save_img:
                        self.draw_tracks_actions(img_orig, gt_tracks, frame_idx)
                        self.save_drawing(path, img_orig, vid_cap)
                self.save_frames_tracks_history(gt_tracks, frame_idx)

            elif frame_idx in self.key_frames:
                # print(frame_idx)
                # img_orig = img_dct[frame_idx]

                det_crops_person, det_boxes_person = self.cache.fetch(frame_idx, is_person=True)
                valid_idx = ~np.all(det_boxes_person == 0, axis=1)
                det_boxes_person = det_boxes_person[valid_idx, ...]
                det_crops_person = det_crops_person[valid_idx, ...]

                det_crops_ball, det_boxes_ball = self.cache.fetch(frame_idx, is_person=False)
                valid_idx = ~np.all(det_boxes_ball == 0, axis=1)
                det_boxes_ball = det_boxes_ball[valid_idx, ...]
                det_crops_ball = det_crops_ball[valid_idx, ...]

                person_detections = self.wrapup_detections(det_boxes_person, det_crops_person, is_person=True)
                ball_detections = self.wrapup_detections(det_boxes_ball, det_crops_ball, is_person=False)
                pred_tracks = self.deepsort.update(person_detections, ball_detections, img_orig)

                if len(pred_tracks) > 0:
                    track_ids = pred_tracks[:, 4]
                    pred_tracks[:, 4] = [int(self.track_gt_idMap[id]) if id in self.track_gt_idMap else id for id in track_ids]
                    self.online_action_detector.update_catches(pred_tracks, frame_idx)
                    if self.opt.save_img:
                        self.draw_tracks_actions(img_orig, pred_tracks, frame_idx)
                        self.save_drawing(path, img_orig, vid_cap)

                self.save_frames_tracks_history(pred_tracks, frame_idx)

        # self.detect_action_online(outpath)
        self.detect_action_offline(outpath)
        # self.save_offline_detection(self.tracks_history, self.unmatched_tracks_history, self.unmatched_detections_history, self.frames_idx_history)

    def wrapup_detections(self, boxes, crops, is_person):
        def get_features(crops):
            features = self.extractor(crops)

            return features

        def _xyxy_to_tlwh(bbox_xyxy):
            if isinstance(bbox_xyxy, np.ndarray):
                bbox_tlwh = bbox_xyxy.copy()
            elif isinstance(bbox_xyxy, torch.Tensor):
                bbox_tlwh = bbox_xyxy.clone()
            bbox_tlwh[:, 2:4] = bbox_xyxy[:, 2:4] - bbox_xyxy[:, :2]
            return bbox_tlwh

        features = get_features(crops)
        bbox_tlwh = _xyxy_to_tlwh(boxes)

        if is_person:
            detections = [Detection(bbox_tlwh[i], features[i], 0) for i in range(bbox_tlwh.shape[0])]
        else:
            detections = [Detection(bbox_tlwh[i], features[i], 1) for i in range(bbox_tlwh.shape[0])]

        return detections

    def get_crops(self, dets, ori_img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64), interpolation=InterpolationMode.BICUBIC)
        ])

        bboxes_ltrb, confs, clses = dets[:, :4], dets[:, 4], dets[:, 5]
        im_crops = []
        for box in bboxes_ltrb:
            x1, y1, x2, y2 = box
            im = ori_img[int(y1):int(y2), int(x1):int(x2)]
            im_crops.append(np.array(transform(im)))
        return im_crops

    def save_online_detection(self):
        save_path = os.path.join(self.opt.output, os.path.basename(self.opt.source)[:-4], 'online_action_detection')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_pkl(self.online_action_detector.history_collision_summary, os.path.join(save_path, 'history_collision_summary.pkl'))
        save_pkl(self.online_action_detector.latest_bp_assoc_dct, os.path.join(save_path, 'latest_bp_assoc_dct.pkl'))

    def save_offline_detection(self, tracks_history, unmatched_tracks_history, unmatched_detections_history, frames_idx_history):
        save_path = os.path.join(self.opt.output, os.path.basename(self.opt.source)[:-4], 'offline_action_detection')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_pkl(tracks_history, os.path.join(save_path, 'tracks_history.pkl'))
        save_pkl(frames_idx_history, os.path.join(save_path, 'frames_idx_history.pkl'))

    def detect_action_offline(self, outpath):
        self.offline_action_detector = OfflineActionDetector(self.tracks_history, self.frames_idx_history,
                                                             ball_ids=self.gt_bids, person_ids=self.gt_pids)
        self.offline_action_detector.write_catches(outpath)

    def detect_action_online(self, outpath):
        self.online_action_detector.write_catches(outpath)

    # def draw_tracks_actions(self, img_orig, tracks, ball_detect, frame_idx):
    def draw_tracks_actions(self, img_orig, tracks, frame_idx):
        def compute_color_for_labels(label):
            """
            Simple function that adds fixed color depending on the class
            """
            palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
            color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
            return tuple(color)

        def draw_boxes(img, tracks, offset=(0, 0)):
            # bboxes_xyxy, ids, clses, scores = tracks[:, :4], tracks[:, 4], tracks[:, 5], tracks[:, 6]
            bboxes_xyxy, ids, clses = tracks[:, :4], tracks[:, 4], tracks[:, 5]

            img_h, img_w, _ = img.shape
            status_scale, id_scale_ball, id_scale_person = 1080, 270, 135
            for i, box in enumerate(bboxes_xyxy):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]

                id = int(ids[i])

                if clses[i] == ObjectCategory.BALL.value:
                    id_size = img_h // id_scale_ball
                    box_color = [0, 255, 0]  # Green
                    id_color = [0, 255, 255] # Yellow
                else:
                    id_size = img_h // id_scale_person
                    box_color = [255, 0, 0]  # Blue
                    id_color = [0, 0, 255]   # Red

                cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=3)
                id_text_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_PLAIN, fontScale=id_size, thickness=4)[0]
                textX, textY = x1 + ((x2 - x1) - id_text_size[0]) // 2, y1 + ((y2 - y1) + id_text_size[1]) // 2
                cv2.putText(img, text=str(id), org=(textX, textY), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=id_size, color=id_color, thickness=4)

        def draw_frame_idx(img, frame_idx):
            framestr = 'Frame {frame}'
            text = framestr.format(frame=frame_idx)
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.putText(img, text, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        draw_boxes(img_orig, tracks)
        draw_frame_idx(img_orig, frame_idx)

    def draw_det_bboxes(self, img_orig, dets, frame_idx):
        def draw_boxes(img, dets, offset=(0, 0)):
            bboxes_xyxy, scores, clses = dets[:, :4], dets[:, 4], dets[:, 5]
            img_h, img_w, _ = img.shape
            text_scale = 540
            for i, box in enumerate(bboxes_xyxy):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]

                score_text = '%d%%' % int(scores[i]*100)
                text_size = img_h // text_scale
                score_text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_PLAIN, fontScale=text_size, thickness=2)[0]
                cv2.rectangle(img, (x1, y1), (x1 + score_text_size[0] + 1, y1 + score_text_size[1] + 1), color=[0, 0, 0], thickness=-1)
                cv2.putText(img, text=score_text, org=(x1, y1 + score_text_size[1] + 1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=text_size, color=[255, 255, 255], thickness=2)

                if clses[i] == 'sports ball':
                    box_color = [0, 255, 0]  # Green
                else:
                    box_color = [255, 0, 0]  # Blue

                cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=3)

        def draw_frame_idx(img, frame_idx):
            framestr = 'Frame {frame}'
            text = framestr.format(frame=frame_idx)
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.putText(img, text, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        draw_boxes(img_orig, dets)
        draw_frame_idx(img_orig, frame_idx)

    def save_drawing(self, path, img_orig, vid_cap):
        save_path = str(Path(self.opt.output) / Path(path).name)
        if self.vid_path != save_path:  # new video
            self.vid_path = save_path
            if isinstance(self.vid_writer, cv2.VideoWriter):
                self.vid_writer.release()  # release previous video writer

            # fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            #     vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, w, h = 60, 1920, 1080
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.opt.fourcc), fps, (w, h))
        self.vid_writer.write(img_orig)


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default=os.path.join(dir_path, 'weights/yolov5s_qnnpack.torchscript'), help='model.pt path')
    parser.add_argument('--source', type=str, default=os.path.join(dir_path, 'inference/images'), help='source')
    parser.add_argument('--output', type=str, default=os.path.join(dir_path, 'outputs'), help='output folder')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument("--config_deepsort", type=str, default=os.path.join(dir_path, "configs/deep_sort.yaml"))
    parser.add_argument('--groundtruths', default=os.path.join(dir_path, '/inputs/groundtruths.txt'), help='path to the groundtruths.txt or \'disable\'')
    parser.add_argument('--save-img', action='store_true', help='save video to outputs')
    parser.add_argument('--debugging', action='store_true', help='switch on the debugging mode')
    parser.add_argument('--skip-frames', type=int, default=10, help='number of frames skipped after each frame scanned')

    return parser


def main(vid_src=None, grd_src=None):
    '''
    Main function that will be ran when performing submission testing.
    We will provide the path of the video and groundtruths when testing.
    argv[1] = video path (--source input)
    argv[2] = groundtruths path (--groundtruths input)
    '''
    parser = default_parser()
    if vid_src == None and grd_src == None:
        vid_src, grd_src = sys.argv[1], sys.argv[2]
    args = parser.parse_args(args=['--source', vid_src, '--groundtruths', grd_src, '--output', './outputs'])

    t0 = time.perf_counter()
    solution = Solution(args)
    with torch.no_grad():
        solution.run()
        # solution.read_csv_gt_tracks()
    t1 = time.perf_counter()
    print('Total Runtime = %.2f' % (t1 - t0))

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    t0 = time.perf_counter()
    solution = Solution(args)
    with torch.no_grad():
        solution.run()
    t1 = time.perf_counter()

    print('Total Runtime = %.2f' % (t1 - t0))