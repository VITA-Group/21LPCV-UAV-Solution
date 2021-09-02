import numpy as np
import torch

from .nn_matching import NearestNeighborDistanceMetric
from .tracker import Tracker

import datetime


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, max_dist=0.2, nn_budget=100):
        max_cosine_distance = max_dist
        metric_person = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        metric_ball = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker_person = Tracker(metric_person)
        self.tracker_ball = Tracker(metric_ball)


    def update(self, person_detections, ball_detections, orig_img):

        self.tracker_person.update(person_detections)

        if ball_detections:
            self.tracker_ball.update(ball_detections)
            tracks = self.tracker_person.tracks + self.tracker_ball.tracks
        else:
            tracks = self.tracker_person.tracks


        # output bbox identities
        active_tracks = []
        for now_line, track in enumerate(tracks):
            if track.is_missed():
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box, orig_img)
            active_tracks.append(np.array([x1, y1, x2, y2, track.track_id, track.cls], dtype=np.int))

        if active_tracks:
            active_tracks = np.stack(active_tracks, axis=0)

        return active_tracks


    def _tlwh_to_xyxy(self, bbox_tlwh, img):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        height, width = img.shape[:2]
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),height-1)
        return x1,y1,x2,y2

