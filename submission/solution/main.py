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
from utils.experimental import (save_pkl, load_pkl, init_pd_csv_reader, load_selected_labels_csv)
from utils.draw_tool import DrawTool
from utils.detection import Detection
from utils.parser import get_config
from utils.box import Box

from deep_assoc import DeepAssoc
from deep_assoc.feature_extractor import Extractor



from modules.baseline_action_detector import BaselineActionDetector
from modules.improved_action_detector import ImprovedActionDetector
from modules.activity_region_cropper import ActivityRegionCropper
from modules.crops_boxes_cache import CropsBoxesCache
from modules.image_pool import ImagePool

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

torch.backends.quantized.engine = 'qnnpack'


class Solution(object):
    def __init__(self, opt):
        self.opt = opt
        cfg = get_config()
        cfg.merge_from_file(opt.config_file)
        self.cfg = cfg

        # make new output folder
        if not os.path.exists(opt.output):
            os.makedirs(opt.output)

        '''
        Initialize deep association online tracker
        '''

        self.extractor = Extractor(os.path.join(dir_path, self.cfg.DEEPASSOC.REID_CKPT))

        self.deep_assoc = DeepAssoc(max_dist=self.cfg.DEEPASSOC.MAX_DIST, nn_budget=self.cfg.DEEPASSOC.NN_BUDGET)

        '''
        Initialize person/ball detector
        '''
        self.grid = torch.load(os.path.join(dir_path, 'weights/grid.pt'), map_location='cpu')
        self.yolo = torch.jit.load(self.cfg.YOLO.YOLO_CKPT, map_location='cpu')

        '''
        Initialize Dataloader
        '''
        model_stride = torch.tensor([8, 16, 32])
        self.stride = int(model_stride.max())  # model stride
        self.imgsz = check_img_size(self.cfg.YOLO.IMG_SIZE, s=self.stride)  # check img_size w.r.t. model stride
        self.dataset = LoadImages(opt.source)
        fps, w, h = self.dataset.cap.get(cv2.CAP_PROP_FPS), int(self.dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        '''
        Initialize image pool
        '''
        self.pool = ImagePool(pool_size=self.cfg.VIDEO.POOL_SIZE, pool_h=self.cfg.VIDEO.POOL_H, pool_w=self.cfg.VIDEO.POOL_W, img_h=h, img_w=w)

        '''
        Initialize activity region cropper
        '''
        self.activity_region_cropper = ActivityRegionCropper(extra=0.25)

        '''
        Initialize crops boxes cache
        '''
        self.gt_frames, self.gt_labels_history, self.gt_pids, self.gt_bids = self.read_gt_tracks_from_csv(opt.groundtruths, self.cfg.SKIP.SKIP_GT_FRAMES)
        all_frames = np.arange(self.dataset.nframes).tolist()
        self.pred_frames = [idx for idx in all_frames if idx not in self.gt_frames and idx % self.cfg.SKIP.SKIP_PRED_FRAMES == 0]
        self.cache = CropsBoxesCache(self.pred_frames, self.gt_frames, len(self.gt_pids), len(self.gt_bids), self.cfg.DEEPASSOC.CROP_HEIGHT, self.cfg.DEEPASSOC.CROP_WIDTH)


        self.tracks_history, self.frames_idx_history = [], []
        self.unmatched_tracks_history, self.unmatched_detections_history = {}, {}

        if opt.save_img:
            self.img_cache = {}
            save_path = str(Path(self.opt.output) / Path(self.dataset.files[0]).name)
            # self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.cfg.VIDEO.FOURCC), fps, (w, h))
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.cfg.VIDEO.FOURCC), fps, (self.cfg.VIDEO.POOL_W, self.cfg.VIDEO.POOL_H))

    def check_bp_collision(self, ball_dets, person_dets):
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

    def read_gt_tracks_from_csv(self, filename, sample_rate):
        csv_file_data = init_pd_csv_reader(filename)
        unique_frames = np.sort(np.unique(csv_file_data["Frame"].to_numpy())).tolist()
        unique_ids = np.sort(np.unique(csv_file_data["ID"].to_numpy()))

        gt_labels = np.zeros((len(unique_frames), unique_ids.size, 6), dtype=np.float32)

        gt_pids, gt_bids = [], []
        for i, frame_idx in enumerate(unique_frames):
            labels = load_selected_labels_csv(csv_file_data, frame_number=frame_idx)
            gt_labels[i, :labels.shape[0], :] = labels
            gt_bids += labels[labels[:,0] == 1, 1].astype(int).tolist()
            gt_pids += labels[labels[:,0] == 0, 1].astype(int).tolist()
        return [idx for idx in unique_frames if idx % sample_rate == 0], gt_labels, np.unique(gt_pids).tolist(), np.unique(gt_bids).tolist()

    def update_gt_cache(self, gts, img, gt_ids, order, frame_idx, is_person):
        crops = self.get_crops(gts, img)
        ordered_crops = np.zeros((len(gt_ids), self.cfg.DEEPASSOC.CROP_HEIGHT, self.cfg.DEEPASSOC.CROP_WIDTH, 3), dtype=np.uint8)
        ordered_boxes = np.zeros((len(gt_ids), 4), dtype=np.int32)
        for i, idx in enumerate(order):
            ordered_crops[idx] = crops[i]
            ordered_boxes[idx] = gts[i, :4]
        self.cache.update(frame_idx, ordered_crops, ordered_boxes, is_person=is_person)

    def update_pred_cache(self, dets, img, frame_idx, max_num, is_person):
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
        frame_idx = 0
        usable_frames = sorted(self.gt_frames + self.pred_frames)
        for path, img_orig, vid_cap in self.dataset:
            if frame_idx in usable_frames:
                self.pool.write(img_orig, frame_idx)
                if not self.pool.is_full() and frame_idx != usable_frames[-1]:
                    frame_idx += 1
                    continue
                else:
                    frame_idx += 1
            else:
                frame_idx += 1
                continue

            while not self.pool.is_empty():
                idx, img_resized = self.pool.read()
                img_h, img_w, _ = img_resized.shape  # get image shape
                if idx in self.gt_frames:
                    if self.opt.save_img:
                        self.img_cache[idx] = np.copy(img_resized)
                    gts_raw = self.gt_labels_history[self.gt_frames.index(idx)]
                    # Remove empty annotation with zeros as placeholder
                    gts_raw = gts_raw[~np.all(gts_raw == 0, axis=1)]
                    gts_scaled = np.multiply(gts_raw, np.array([1.0, 1.0, img_w, img_h, img_w, img_h]))
                    gts_ltrb = Box.cxcywh_to_ltrb(gts_scaled[:, 2:], img_resized)
                    # ltrb, id, cls
                    gts = np.concatenate((gts_ltrb, gts_scaled[:, 1:2], gts_scaled[:, 0:1]), 1).astype(int)
                    gts_person, gts_ball = np.copy(gts[gts[:, 5] == 0]), np.copy(gts[gts[:, 5] == 1])

                    self.activity_region_cropper.update_memory(gts[:, :4])

                    gts_person_ids = gts_person[:, 4]
                    persons_order = [self.gt_pids.index(pid) for pid in gts_person_ids]
                    gts_ball_ids = gts_ball[:, 4]
                    balls_order = [self.gt_bids.index(bid) for bid in gts_ball_ids]

                    if gts_person.size > 0:
                        self.update_gt_cache(gts_person, img_resized, self.gt_pids, persons_order, idx, is_person=True)
                    if gts_ball.size > 0:
                        self.update_gt_cache(gts_ball, img_resized, self.gt_bids, balls_order, idx, is_person=False)
                elif idx in self.pred_frames:
                    if self.opt.save_img:
                        self.img_cache[idx] = np.copy(img_resized)
                    '''
                    Crop the activity region from the latest prediction
                    (x_min, y_min) and (x_max, y_max) are the coords of the minimum bounding rectangle
                    '''
                    latest_det = self.activity_region_cropper.latest_det
                    img_crop, dx, dy = self.activity_region_cropper.crop_image(img_resized, latest_det)
                    img_crop_letterbox = self.activity_region_cropper.letterbox_image(img_crop, imgsz=self.imgsz, stride=self.stride)

                    preds = self.yolo(img_crop_letterbox)
                    preds = postprocess(preds, self.grid, self.yolo)

                    # Apply NMS
                    # ltrb, conf, cls
                    dets = non_max_suppression(preds[0], self.cfg.YOLO.CONF_THRESH, self.cfg.YOLO.IOU_THRESH, classes=None, agnostic=False)[0]

                    if dets.numel() == 0:
                        self.pred_frames.remove(idx)
                        continue

                    dets = dets.detach().numpy()

                    dets_person = np.copy(dets[dets[:, 5] == 0])
                    dets_ball = np.copy(dets[dets[:, 5] == 1])
                    if dets_person.shape[0] != 0 and dets_ball.shape[0] != 0:
                        max_ball_height = (dets_person[:, 3] - dets_person[:, 1]).mean(axis=0) * 0.6
                        dets_ball = dets_ball[(dets_ball[:, 3] - dets_ball[:, 1]) < max_ball_height]
                        dets = np.concatenate((dets_person, dets_ball), axis=0)

                    dets = self.activity_region_cropper.coords_final2orig(dets, img_crop_letterbox, img_crop, dx, dy)
                    self.activity_region_cropper.update_memory(dets[:, :4])

                    # if len(clses_person) < gts_persons.shape[0]:
                    self.activity_region_cropper.set_extra_ratio(0.1)
                    # else:
                    #     self.activity_region_cropper.set_extra_ratio(0.2)

                    dets_person, dets_ball = dets[dets[:,5]==0,:], dets[dets[:,5]==1,:]

                    if dets_ball.size == 0 or dets_person.size == 0:
                        self.pred_frames.remove(idx)
                        continue

                    bp_collistion_matx = self.check_bp_collision(dets_ball, dets_person).reshape(
                                                                        (dets_ball.shape[0], dets_person.shape[0]))
                    if not np.any(bp_collistion_matx):
                        self.pred_frames.remove(idx)
                        continue

                    if dets_person.size > 0:
                        self.update_pred_cache(dets_person, img_resized, idx, len(self.gt_pids), is_person=True)
                    if dets_ball.size > 0:
                        self.update_pred_cache(dets_ball, img_resized, idx, len(self.gt_bids), is_person=False)
            self.pool.reset()

        gt_btracks_dct, gt_ptracks_dct = {}, {}
        gt_bdets_dct, gt_pdets_dct = {}, {}
        for frame_idx in range(self.dataset.nframes):
            if frame_idx in self.gt_frames:
                gt_crops_person, gt_boxes_person, valid_idx = self.cache.fetch(frame_idx, is_person=True)
                selected_pids = np.array(self.gt_pids)[valid_idx]

                gt_crops_ball, gt_boxes_ball, valid_idx = self.cache.fetch(frame_idx, is_person=False)
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

        self.deep_assoc.tracker_person.initiate_tracks(gt_ptracks_dct, 0)
        self.deep_assoc.tracker_ball.initiate_tracks(gt_btracks_dct, 1)

        img_h, img_w = self.pool.get_pool_hw()
        for frame_idx in range(self.dataset.nframes):
            if frame_idx in self.gt_frames:
                person_detections, ball_detections = gt_pdets_dct[frame_idx], gt_bdets_dct[frame_idx]
                gt_tracks = self.deep_assoc.update(person_detections, ball_detections, img_h, img_w)

                if len(gt_tracks) > 0:
                    self.save_tracks(gt_tracks, frame_idx)
                    if self.opt.save_img:
                        img_resized = self.img_cache[frame_idx]
                        bboxes_ltrb, ids, clses = gt_tracks[:, :4], gt_tracks[:, 4], gt_tracks[:, 5]
                        DrawTool.draw_tracks(img_resized, bboxes_ltrb, ids, clses)
                        DrawTool.draw_frame_idx(img_resized, frame_idx, 'Groundtruth')
                        self.vid_writer.write(img_resized)

            elif frame_idx in self.pred_frames:
                det_crops_person, det_boxes_person, _ = self.cache.fetch(frame_idx, is_person=True)
                det_crops_ball, det_boxes_ball, _ = self.cache.fetch(frame_idx, is_person=False)

                person_detections = self.wrapup_detections(det_boxes_person, det_crops_person, is_person=True)
                ball_detections = self.wrapup_detections(det_boxes_ball, det_crops_ball, is_person=False)
                pred_tracks = self.deep_assoc.update(person_detections, ball_detections, img_h, img_w)

                if len(pred_tracks) > 0:
                    self.save_tracks(pred_tracks, frame_idx)
                    if self.opt.save_img:
                        img_resized = self.img_cache[frame_idx]
                        bboxes_ltrb, ids, clses = pred_tracks[:, :4], pred_tracks[:, 4], pred_tracks[:, 5]
                        DrawTool.draw_tracks(img_resized, bboxes_ltrb, ids, clses)
                        DrawTool.draw_frame_idx(img_resized, frame_idx, 'Prediction')
                        self.vid_writer.write(img_resized)

        self.detect_action_baseline()
        self.detect_action_improved()
        self.save_tracks_history_to_disk(self.tracks_history, self.frames_idx_history, self.gt_bids, self.gt_pids)

    def wrapup_detections(self, boxes, crops, is_person):
        features = self.extractor(crops)
        bbox_ltwh = Box.ltrb_to_ltwh(boxes)

        if is_person:
            detections = [Detection(bbox_ltwh[i], features[i], 0) for i in range(bbox_ltwh.shape[0])]
        else:
            detections = [Detection(bbox_ltwh[i], features[i], 1) for i in range(bbox_ltwh.shape[0])]

        return detections

    def get_crops(self, dets, ori_img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.cfg.DEEPASSOC.CROP_HEIGHT, self.cfg.DEEPASSOC.CROP_WIDTH), interpolation=InterpolationMode.BICUBIC)
        ])

        bboxes_ltrb, confs, clses = dets[:, :4], dets[:, 4], dets[:, 5]
        im_crops = []
        for box in bboxes_ltrb:
            x1, y1, x2, y2 = box
            im = ori_img[int(y1):int(y2), int(x1):int(x2)]
            im_crops.append(np.array(transform(im)))
        return im_crops

    def save_tracks_history_to_disk(self, tracks_history, frames_idx_history, ball_ids, person_ids):
        save_path = os.path.join(self.opt.output, os.path.basename(self.opt.source)[:-4], 'tracking')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_pkl(tracks_history, os.path.join(save_path, 'tracks_history.pkl'))
        save_pkl(frames_idx_history, os.path.join(save_path, 'frames_idx_history.pkl'))
        save_pkl(ball_ids, os.path.join(save_path, 'ball_ids.pkl'))
        save_pkl(person_ids, os.path.join(save_path, 'person_ids.pkl'))

    def save_tracks(self, tracks, frame_idx):
        # tracks_cxcywh[:, :2] = 0.5 * (tracks[:, :2] + tracks[:, 2:4])
        tracks_ltrb = np.zeros_like(tracks[:, :6], dtype=int)
        tracks_ltrb[:, :4] = tracks[:, :4]
        # tracks_ltwh[:, 2:4] = tracks[:, 2:4] - tracks[:, :2]
        tracks_ltrb[:, 4:6] = tracks[:, 4:6]

        self.tracks_history.append(tracks_ltrb)
        self.frames_idx_history.append(frame_idx)

    def detect_action_improved(self):
        outpath = os.path.basename(self.opt.source)[:-4]
        outpath = os.path.join(self.opt.output, outpath + '_improved_out.csv')
        action_detector = ImprovedActionDetector(self.tracks_history, self.frames_idx_history,
                                                 ball_ids=self.gt_bids, person_ids=self.gt_pids)
        action_detector.write_catches(outpath)

    def detect_action_baseline(self):
        outpath = os.path.basename(self.opt.source)[:-4]
        outpath = os.path.join(self.opt.output, outpath + '_baseline_out.csv')
        action_detector = BaselineActionDetector(self.tracks_history, self.frames_idx_history,
                                                 ball_ids=self.gt_bids, person_ids=self.gt_pids)
        action_detector.write_catches(outpath)


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=os.path.join(dir_path, 'inputs/7p3b_02M/7p3b_02M.m4v'), help='source')
    parser.add_argument('--output', type=str, default=os.path.join(dir_path, 'outputs'), help='output folder')
    parser.add_argument('--groundtruths', default=os.path.join(dir_path, 'inputs/7p3b_02M/7p3b_02M_init.csv'), help='path to the groundtruths.txt or \'disable\'')
    parser.add_argument("--config_file", type=str, default=os.path.join(dir_path, "configs.yaml"))
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--save-img', action='store_true', help='save video to outputs')
    parser.add_argument('--debugging', action='store_true', help='switch on the debugging mode')

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
        # solution.read_gt_tracks_from_csv()
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