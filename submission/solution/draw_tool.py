import cv2
from enums import ObjectCategory


class DrawTool(object):
    @staticmethod
    def draw_frame_idx(img, frame_idx):
        framestr = 'Frame {frame}'
        text = framestr.format(frame=frame_idx)
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.putText(img, text, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    @staticmethod
    def compute_color_for_labels(label):
        """
        Simple function that adds fixed color depending on the class
        """
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    @staticmethod
    def draw_tracks(img, tracks, frame_idx):

        # bboxes_xyxy, ids, clses, scores = tracks[:, :4], tracks[:, 4], tracks[:, 5], tracks[:, 6]
        bboxes_xyxy, ids, clses = tracks[:, :4], tracks[:, 4], tracks[:, 5]

        img_h, img_w, _ = img.shape
        status_scale, id_scale_ball, id_scale_person = 1080, 270, 135
        for i, box in enumerate(bboxes_xyxy):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            id = int(ids[i])
            if clses[i] == ObjectCategory.BALL.value:
                id_size = img_h // id_scale_ball
                box_color = [0, 255, 0]  # Green
                id_color = [0, 255, 255] # Yellow
            else:
                id_size = img_h // id_scale_person
                box_color = [255, 0, 0]  # Blue
                id_color = [0, 0, 255]   # Red
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=3)
            id_text_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_PLAIN, fontScale=id_size, thickness=4)[0]
            textX, textY = x1 + ((x2 - x1) - id_text_size[0]) // 2, y1 + ((y2 - y1) + id_text_size[1]) // 2
            cv2.putText(img, text=str(id), org=(textX, textY), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=id_size, color=id_color, thickness=4)
        DrawTool.draw_frame_idx(img, frame_idx)

    @staticmethod
    def draw_detections(img, dets, frame_idx):
        bboxes_xyxy, scores, clses = dets[:, :4], dets[:, 4], dets[:, 5]
        img_h, img_w, _ = img.shape
        text_scale = 540
        for i, box in enumerate(bboxes_xyxy):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            score_text = '%d%%' % int(scores[i]*100)
            text_size = img_h // text_scale
            score_text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_PLAIN, fontScale=text_size, thickness=2)[0]
            cv2.rectangle(img, (x1, y1), (x1 + score_text_size[0] + 1, y1 + score_text_size[1] + 1), color=[0, 0, 0], thickness=-1)
            cv2.putText(img, text=score_text, org=(x1, y1 + score_text_size[1] + 1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=text_size, color=[255, 255, 255], thickness=2)
            if clses[i] == 'sports ball':
                box_color = [0, 255, 0]  # Green
            else:
                box_color = [255, 0, 0]  # Blue
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=3)
        DrawTool.draw_frame_idx(img, frame_idx)
