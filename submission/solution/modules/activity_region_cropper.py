from utils.datasets import letterbox
from utils.general import scale_coords
import torch
import numpy as np

class ActivityRegionCropper(object):
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
