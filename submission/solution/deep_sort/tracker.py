# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from .linear_assignment import min_cost_matching
from .track import Track
from .detection import Detection
import torch

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    # def __init__(self, metric, target, max_iou_distance=0.7):
    def __init__(self, metric):
        self.metric = metric

        self.tracks = []
        self._next_id = 1
        self.track_limit = 0  # Maximum number of tracks
        self.track_id_start = 0  # The start id of tracks
        self.track_count = 0 # Counter on the number of created tracks
        # self.target = target

    def set_track_limit(self, limit):
        self.track_limit = limit


    def set_track_start(self, start):
        self.track_id_start = start


    # def update(self, detections, img, extractor):
    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # detections = self.wrapup_detections(detections, img, extractor)

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # if self.track_count < self.track_limit:
        #     for detection_idx in unmatched_detections:
        #         self._initiate_track(detections[detection_idx])
        #         self.track_count += 1

        # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks]
        features, targets = [], []
        for track in self.tracks:
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets))

        return [self.tracks[idx] for idx in unmatched_tracks], [detections[idx] for idx in unmatched_detections]


    def _match(self, detections):

        def distance_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return cost_matrix

        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections = \
            min_cost_matching(
                distance_metric, self.metric.matching_threshold,
                self.tracks, detections)

        return matches, unmatched_tracks, unmatched_detections


    def _initiate_track(self, detection):
        mean = detection.to_xyah()
        cls = detection.clses
        self.tracks.append(Track(
            mean, self.track_id_start + self._next_id, cls, detection.feature))
        self._next_id += 1
        return self._next_id-1


    def wrapup_detections(self, dets, orig_img, extractor):
        bboxes_ltrb, confs, clses = dets[:, :4], dets[:, 4], dets[:, 5]
        features = self._get_features(bboxes_ltrb, orig_img, extractor)
        bbox_tlwh = self._xyxy_to_tlwh(bboxes_ltrb)
        detections = [Detection(bbox_tlwh[i], conf, features[i], clses[i]) for i,conf in enumerate(confs)]

        scores = np.array([d.confidence for d in detections])
        indices = np.flip(np.argsort(scores))
        detections = [detections[i] for i in indices]

        return detections


    def _get_features(self, bboxes_ltrb, ori_img, extractor):
        im_crops = []
        for box in bboxes_ltrb:
            # x1,y1,x2,y2 = self._xywh_to_xyxy(box, ori_img)
            x1,y1,x2,y2 = box
            im = ori_img[int(y1):int(y2), int(x1):int(x2)]
            im_crops.append(im)
        if im_crops:
            features = extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _xyxy_to_tlwh(self, bbox_xyxy):
        # x1,y1,x2,y2 = bbox_xyxy
        #
        # t = x1
        # l = y1
        # w = int(x2-x1)
        # h = int(y2-y1)
        # return t,l,w,h
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:,2:4] = bbox_xyxy[:,2:4] - bbox_xyxy[:,:2]
        return bbox_tlwh



    def initiate_tracks(self, gt_tracks_dct, cls):
        for id, detections in gt_tracks_dct.items():
            det_init = detections[0]
            mean = det_init.to_xyah()
            self.tracks.append(Track(
                mean, id, cls, det_init.feature))

        features, targets = [], []
        for track in self.tracks:
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets))

        features, targets = [], []
        for track in self.tracks:
            detections = gt_tracks_dct[track.track_id]
            for det in detections:
                features.append(det.feature)
                targets.append(track.track_id)
            # track.features = []
        self.metric.init_cache(
            np.asarray(features), np.asarray(targets))

        # for detection_idx in unmatched_detections:
        #     self._initiate_track(detections[detection_idx])
        #     self.track_count += 1