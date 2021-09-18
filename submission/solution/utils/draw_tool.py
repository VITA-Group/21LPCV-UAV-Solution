import cv2
from .enums import ObjectCategory
import numpy as np

class DrawTool(object):
    @staticmethod
    def draw_frame_idx(img, frame_idx, extra_info):
        framestr = '{extra} Frame {frame}'
        text = framestr.format(extra=extra_info, frame=frame_idx)
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 24, 24)[0]
        cv2.putText(img, text, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 4, [255, 255, 255], 2)

    @staticmethod
    def compute_color_for_labels(label):
        """
        Simple function that adds fixed color depending on the class
        """
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    @staticmethod
    def draw_tracks(img, bboxes_ltrb, ids, clses):
        img_h, img_w, _ = img.shape
        id_scale_ball, id_scale_person = 270, 135
        for i, box in enumerate(bboxes_ltrb):
            l, t, r, b = [int(coord) for coord in box]
            id = int(ids[i])
            if clses[i] == 1:
                id_size = img_h // id_scale_ball
                box_color = [0, 255, 0]  # Green
                id_color = [0, 255, 255]  # Yellow
            else:
                id_size = img_h // id_scale_person
                box_color = [255, 0, 0]  # Blue
                id_color = [0, 0, 255]  # Red
            cv2.rectangle(img, (l, t), (r, b), color=box_color, thickness=3)

            id_text_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_PLAIN, fontScale=id_size, thickness=4)[0]
            textX, textY = l + (r - l - id_text_size[0]) // 2, t + (b - t + id_text_size[1]) // 2
            cv2.putText(img, text=str(id), org=(textX, textY), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=id_size, color=id_color, thickness=4)

    @staticmethod
    def draw_detections(img, bboxes_ltrb, confs, clses):
        img_h, img_w, _ = img.shape
        text_scale = 540

        for i, box in enumerate(bboxes_ltrb):
            l, t, r, b = [int(coord) for coord in box]
            if clses[i] == 1:
                box_color = [0, 255, 0]
            else:
                box_color = [255, 0, 0]
            cv2.rectangle(img, (l, t), (r, b), color=box_color, thickness=3)
            score_text = '%d%%' % int(confs[i] * 100)
            text_size = img_h // text_scale
            score_text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_PLAIN, fontScale=text_size, thickness=2)[0]
            cv2.rectangle(img, (l, t), (l + score_text_size[0] + 1, t + score_text_size[1] + 1), color=[0, 0, 0],
                          thickness=-1)
            cv2.putText(img, text=score_text, org=(l, t + score_text_size[1] + 1), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=text_size, color=[255, 255, 255], thickness=2)

    @staticmethod
    def draw_bp_assoc(img, tracks, prev_assocs, current_assocs):
        clses, ids = tracks[:, 0], tracks[:, 1]
        img_h, img_w, _ = img.shape

        wrapped_text = [
            ' '.join([str(assoc[1]) for assoc in prev_assocs]),
            ' '.join([str(assoc[1]) for assoc in current_assocs])
        ]
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]

            gap = textsize[1] + 5

            y = int((img.shape[0] + textsize[1]) / 2) + i * gap
            x = 10  # for center alignment => int((img.shape[1] - textsize[0]) / 2)

            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4, lineType=cv2.LINE_AA)
        for bid, pid in current_assocs:
            if not np.any(ids == bid) or not np.any(ids == pid):
                continue
            btrack = tracks[ids == bid, :].squeeze()
            assert btrack[0] == 1
            bbox_center = 0.5 * (btrack[2:4] + btrack[4:6])

            ptrack = tracks[ids == pid, :].squeeze()
            assert ptrack[0] == 0
            pbox_center = 0.5 * (ptrack[2:4] + ptrack[4:6])

            cv2.line(img, bbox_center.astype(int), pbox_center.astype(int), color=(0, 0, 255), thickness=20)

    @staticmethod
    def draw_unmatched_errors(img, unmatched_tracks, unmatched_detects):
        if len(unmatched_tracks) > 0:
            track_bboxes_ltrb, track_ids, track_clses = unmatched_tracks[:, :4], unmatched_tracks[:, 4], unmatched_tracks[:, 5]
            DrawTool.draw_tracks(img, track_bboxes_ltrb, track_ids, track_clses)
        if len(unmatched_detects) > 0:
            det_bboxes_ltrb, det_confs, det_clses = unmatched_detects[:, :4], unmatched_detects[:, 4], unmatched_detects[:, 5]
            DrawTool.draw_detections(img, det_bboxes_ltrb, det_confs, det_clses)

    @staticmethod
    def draw_reid_errors(img, pred_tracks, gt_tracks):
        def compute_iou(pred_box, gt_box):
            """
            pred_box : the coordinate for predict bounding box
            gt_box :   the coordinate for ground truth bounding box
            return :   the iou score
            """
            # 1.get the coordinate of inters
            x1_min, y1_min, x1_max, y1_max = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
            x2_min, y2_min, x2_max, y2_max = gt_box[0], gt_box[1], gt_box[2], gt_box[3]

            xmin = max(x1_min, x2_min)
            ymin = max(y1_min, y2_min)
            xmax = min(x1_max, x2_max)
            ymax = min(y1_max, y2_max)

            w = np.maximum(xmax - xmin + 1., 0.)
            h = np.maximum(ymax - ymin + 1., 0.)

            # 2. calculate the area of inters
            inters = w * h

            # 3. calculate the area of union
            union = ((x1_max - x1_min + 1.) * (y1_max - y1_min + 1.) +
                   (x2_max - x2_min + 1.) * (y2_max - y2_min + 1.) -
                   inters)

            # 4. calculate the overlaps between pred_box and gt_box
            iou = inters / union

            return iou

        pred_clses, pred_ids, pred_bboxes_ltrb = pred_tracks[:, 0], pred_tracks[:, 1], pred_tracks[:, 2:6]
        gt_clses, gt_ids, gt_bboxes_ltrb = gt_tracks[:, 0], gt_tracks[:, 1], gt_tracks[:, 2:6]

        id_iou_dct = {}
        for i, gt_id in enumerate(gt_ids):
            gt_box = gt_bboxes_ltrb[i, :].squeeze()
            pred_box = pred_bboxes_ltrb[pred_ids == gt_id, :].squeeze()
            if pred_box.size == 0:
                iou = 0
            else:
                iou = compute_iou(pred_box, gt_box)
            id_iou_dct[gt_id] = iou
        iou_thresh = 0.5

        error_ids = []
        for id, iou in id_iou_dct.items():
            if iou < iou_thresh:
                error_ids.append(id)

        img_h, img_w, _ = img.shape
        id_scale_ball, id_scale_person = 270, 135
        print(error_ids)
        print(gt_ids)
        print(pred_ids)
        for id in error_ids:
            select = gt_ids == id
            box = gt_bboxes_ltrb[select, :].squeeze()
            print(box)
            l, t, w, h = [int(coord) for coord in box]
            cls = gt_clses[select].squeeze()
            if cls == 1:
                id_size = img_h // id_scale_ball
                box_color = [0, 255, 0]  # Green
                id_color = [0, 255, 255]  # Yellow
            else:
                id_size = img_h // id_scale_person
                box_color = [255, 0, 0]  # Blue
                id_color = [0, 0, 255]  # Red
            cv2.rectangle(img, (l, t), (l+w, t+h), color=box_color, thickness=3)

            id_text_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_PLAIN, fontScale=id_size, thickness=4)[0]
            textX, textY = l + (w - id_text_size[0]) // 2, t + (h + id_text_size[1]) // 2
            cv2.putText(img, text=str(id), org=(textX, textY), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=id_size, color=id_color, thickness=4)