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
        # self.target = target


    # def update(self, detections, img, extractor):
    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

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