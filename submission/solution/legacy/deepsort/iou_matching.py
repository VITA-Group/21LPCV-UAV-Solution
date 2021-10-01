# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np

def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def IoU(bboxes, candidates, epsilon=1e-5):
    # pairwise jaccard botween boxes a and boxes b
    # box: [top, left, bottom, right]
    tl = np.maximum(bboxes[:, np.newaxis, :2], candidates[:, :2])
    br = np.minimum(bboxes[:, np.newaxis, 2:], candidates[:, 2:])
    inter = np.clip(br - tl, a_min=0, a_max=None)

    area_int = np.prod(inter, axis=2)
    area_bboxes = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
    area_candidates = np.prod(candidates[:, 2:] - candidates[:, :2], axis=1)

    area_union = area_bboxes[:, np.newaxis] + area_candidates - area_int
    return area_int / np.clip(area_union, a_min=epsilon, a_max=None)  # shape: (len(a) x len(b))

def DIoU(bboxes, candidates):

    inter_br = np.minimum(bboxes[:, np.newaxis, 2:], candidates[:, 2:])
    inter_tl = np.maximum(bboxes[:, np.newaxis, :2], candidates[:, :2])
    out_br = np.maximum(bboxes[:, np.newaxis, 2:], candidates[:, 2:])
    out_tl = np.minimum(bboxes[:, np.newaxis, :2], candidates[:, :2])

    inter = np.clip(inter_br - inter_tl, a_min=0, a_max=None)
    area_inter = np.prod(inter, axis=2)
    area_bboxes = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
    area_candidates = np.prod(candidates[:, 2:] - candidates[:, :2], axis=1)
    area_union = area_bboxes[:, np.newaxis] + area_candidates - area_inter

    center_x1 = (bboxes[:, 2] + bboxes[:, 0]) / 2
    center_y1 = (bboxes[:, 3] + bboxes[:, 1]) / 2
    center_x2 = (candidates[:, 2] + candidates[:, 0]) / 2
    center_y2 = (candidates[:, 3] + candidates[:, 1]) / 2

    diag_inter = (np.subtract(center_x2, center_x1[:, np.newaxis])) ** 2 + \
                 (np.subtract(center_y2, center_y1[:, np.newaxis])) ** 2
    outer = np.clip(out_br - out_tl, a_min=0, a_max=None)
    diag_outer = (outer[:, :, 0] ** 2) + (outer[:, :, 1] ** 2)
    diag_inter = diag_inter.reshape(diag_outer.shape)
    dious = area_inter / area_union - diag_inter / diag_outer
    dious = np.clip(dious, a_min=-1.0, a_max=1.0)

    return dious

def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    candidates = np.asarray([detections[idx].to_tlbr() for idx in detection_indices])
    bboxes = np.asarray([tracks[idx].to_tlbr() for idx in track_indices])

    cost_matrix = DIoU(bboxes, candidates)
    if cost_matrix.ndim == 1:
        cost_matrix = cost_matrix[:, np.newaxis]
    cost_matrix = np.multiply(np.ones_like(cost_matrix) - cost_matrix, 0.5)
    return cost_matrix
