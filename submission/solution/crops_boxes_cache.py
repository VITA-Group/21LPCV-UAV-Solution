import numpy as np

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
                self.key_persons_crop_cache[pos][:num_detections] = crops
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
