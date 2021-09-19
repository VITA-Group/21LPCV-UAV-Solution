import numpy as np
import torch

class Box(object):
	@staticmethod
	def ltrb_to_cxcywh(*xyxy):
		"""" Calculates the relative bounding box from absolute pixel values. """
		bbox_left = min([xyxy[0].item(), xyxy[2].item()])
		bbox_top = min([xyxy[1].item(), xyxy[3].item()])
		bbox_w = abs(xyxy[0].item() - xyxy[2].item())
		bbox_h = abs(xyxy[1].item() - xyxy[3].item())
		cx = (bbox_left + bbox_w / 2)
		cy = (bbox_top + bbox_h / 2)
		w = bbox_w
		h = bbox_h
		return int(cx), int(cy), int(w), int(h)

	@staticmethod
	def ltrb_to_ltwh(bboxes_ltrb):
		if isinstance(bboxes_ltrb, np.ndarray):
			bboxes_ltwh = bboxes_ltrb.copy()
		elif isinstance(bboxes_ltrb, torch.Tensor):
			bboxes_ltwh = bboxes_ltrb.clone()
		bboxes_ltwh[:, 2:4] = bboxes_ltrb[:, 2:4] - bboxes_ltrb[:, :2]
		return bboxes_ltwh

	@staticmethod
	def cxcywh_to_ltrb(bboxes_cxcywh, img):
		img_h, img_w, _ = img.shape  # get image shape
		# Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
		bboxes_ltrb = bboxes_cxcywh.clone() if isinstance(bboxes_cxcywh, torch.Tensor) else np.copy(bboxes_cxcywh)
		bboxes_ltrb[:, 0:2] = bboxes_cxcywh[:, 0:2] - bboxes_cxcywh[:, 2:4] / 2  # top left
		bboxes_ltrb[:, 2:4] = bboxes_cxcywh[:, 0:2] + bboxes_cxcywh[:, 2:4] / 2  # bottom right
		bboxes_ltrb[:, 0] = np.maximum(np.minimum(bboxes_ltrb[:, 0], img_w - 1), 0)
		bboxes_ltrb[:, 1] = np.maximum(np.minimum(bboxes_ltrb[:, 1], img_h - 1), 0)
		bboxes_ltrb[:, 2] = np.maximum(np.minimum(bboxes_ltrb[:, 2], img_w - 1), 0)
		bboxes_ltrb[:, 3] = np.maximum(np.minimum(bboxes_ltrb[:, 3], img_h - 1), 0)
		return bboxes_ltrb