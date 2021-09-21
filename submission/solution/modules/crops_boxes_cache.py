import numpy as np

class CropsBoxesCache(object):
    def __init__(self, pred_frames, gt_frames, persons_count, balls_count, crop_height, crop_width):
        self.pred_frames = pred_frames
        self.gt_frames = gt_frames

        self.gt_persons_crop_cache = np.zeros((len(gt_frames), persons_count, crop_height, crop_width, 3), dtype=np.uint8)
        self.gt_balls_crop_cache = np.zeros((len(gt_frames), balls_count, crop_height, crop_width, 3), dtype=np.uint8)

        self.gt_persons_box_cache = np.zeros((len(gt_frames), persons_count, 4), dtype=np.int32)
        self.gt_balls_box_cache = np.zeros((len(gt_frames), balls_count, 4), dtype=np.int32)


        self.pred_persons_crop_cache = np.zeros((len(pred_frames), persons_count, crop_height, crop_width, 3), dtype=np.uint8)
        self.pred_balls_crop_cache = np.zeros((len(pred_frames), balls_count, crop_height, crop_width, 3), dtype=np.uint8)

        self.pred_persons_box_cache = np.zeros((len(pred_frames), persons_count, 4), dtype=np.int32)
        self.pred_balls_box_cache = np.zeros((len(pred_frames), balls_count, 4), dtype=np.int32)


    def update(self, frame_idx, crops, boxes, is_person):
        num_detections = crops.shape[0]
        if frame_idx in self.pred_frames:
            pos = self.pred_frames.index(frame_idx)
            if is_person:
                crop_cache, box_cache = self.pred_persons_crop_cache, self.pred_persons_box_cache
            else:
                crop_cache, box_cache = self.pred_balls_crop_cache, self.pred_balls_box_cache
        elif frame_idx in self.gt_frames:
            pos = self.gt_frames.index(frame_idx)
            if is_person:
                crop_cache, box_cache = self.gt_persons_crop_cache, self.gt_persons_box_cache
            else:
                crop_cache, box_cache = self.gt_balls_crop_cache, self.gt_balls_box_cache
        crop_cache[pos][:num_detections] = crops
        box_cache[pos][:num_detections] = boxes

        return

    def fetch(self, frame_idx, is_person):
        if frame_idx in self.pred_frames:
            pos = self.pred_frames.index(frame_idx)
            if is_person:
                crop_cache, box_cache = self.pred_persons_crop_cache, self.pred_persons_box_cache
            else:
                crop_cache, box_cache = self.pred_balls_crop_cache, self.pred_balls_box_cache
        elif frame_idx in self.gt_frames:
            pos = self.gt_frames.index(frame_idx)
            if is_person:
                crop_cache, box_cache = self.gt_persons_crop_cache, self.gt_persons_box_cache
            else:
                crop_cache, box_cache = self.gt_balls_crop_cache, self.gt_balls_box_cache

        crops, boxes = crop_cache[pos], box_cache[pos]
        valid_idx = ~np.all(boxes == 0, axis=1)
        boxes = boxes[valid_idx, ...]
        crops = crops[valid_idx, ...]

        return crops, boxes, valid_idx
