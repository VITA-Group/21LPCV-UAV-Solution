import cv2
import argparse
import numpy as np
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)
from utils.enums import DivisionType
from utils.experimental import load_labels_from_csv, load_labels_json
from utils.datasets import LoadImages
from utils.draw_tool import DrawTool

class BallReID(object):
    def __init__(self):
        # self.colorOrder = ['red', 'purple', 'blue', 'green', 'yellow', 'orange']
        self.static_clr_offs = (9, 100, 100) # Color tolerance / range for each color (hueOffset, satOffset, valOffset)
        self.dynamic_clr_offs = (2, 50, 50) # Color tolerance for tracking: (hueOffset, satOffset, valOffset)
        self.detected_balls_color_value = {}
        self.static_colorDict, self.rgb_colorDict = self.default_colorDict()
        self.dynamic_colorDict = {}
        self.division_type = DivisionType.GRID.value


    def default_colorDict(self):
        # Color dictonary for ball tracking where red : [(upper), (lower)] in HSV values
        # Static color definition is used when finding the dynamic color dict, or as a fallback.
        # Use https://www.rapidtables.com/web/color/RGB_Color.html for hue

        # Tolerance / range for each color
        hueOffset = self.static_clr_offs[0]
        satOffset = self.static_clr_offs[1]
        valOffset = self.static_clr_offs[2]

        # BGR Values for each color tested
        yellowBGR = np.uint8([[[ 81, 205, 217]]])
        redBGR    = np.uint8([[[ 78,  87, 206]]])
        blueBGR   = np.uint8([[[197, 137,  40]]])
        greenBGR  = np.uint8([[[101, 141,  67]]])
        orangeBGR = np.uint8([[[ 84, 136, 227]]])
        purpleBGR = np.uint8([[[142,  72,  72]]])

        bgr_colorDict = {
            "red":    redBGR,
            "purple": purpleBGR,
            "blue":   blueBGR,
            "green":  greenBGR,
            "yellow": yellowBGR,
            "orange": orangeBGR,
        }

        colorListBGR = [yellowBGR, redBGR, blueBGR, greenBGR, orangeBGR, purpleBGR]
        colorListHSVTmp = []
        colorListHSV = []

        # Convert BGR to HSV
        for bgr in colorListBGR:
            colorListHSVTmp.append(np.squeeze(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)))

        # Create ranges based off offsets
        for i in range(len(colorListBGR)):
            hsv = colorListHSVTmp[i]
            upper = (hsv[0] + hueOffset, hsv[1] + satOffset, hsv[2] + valOffset)
            lower = (hsv[0] - hueOffset, hsv[1] - satOffset, hsv[2] - valOffset)
            colorListHSV.append([upper, lower])

        hsv_colorDict = {
            "red":    [colorListHSV[1][0], colorListHSV[1][1]],
            "purple": [colorListHSV[5][0], colorListHSV[5][1]],
            "blue":   [colorListHSV[2][0], colorListHSV[2][1]],
            "green":  [colorListHSV[3][0], colorListHSV[3][1]],
            "yellow": [colorListHSV[0][0], colorListHSV[0][1]],
            "orange": [colorListHSV[4][0], colorListHSV[4][1]],
        }

        return hsv_colorDict, bgr_colorDict


    def updateDynColorDict(self, gt_tracks, img_orig):
        # Color Dict Functions

        def update_dict():
            hueOffset = self.dynamic_clr_offs[0]
            satOffset = self.dynamic_clr_offs[1]
            valOffset = self.dynamic_clr_offs[2]

            for color, colorVals_id in self.detected_balls_color_value.items():
                # Average color values
                colorVals, id = colorVals_id[0], colorVals_id[1]
                hsv = tuple([int(sum(clmn) / len(colorVals)) for clmn in zip(*colorVals)])

                # Create ranges for color
                upper = (hsv[0] + hueOffset, hsv[1] + satOffset, hsv[2] + valOffset)
                lower = (hsv[0] - hueOffset, hsv[1] - satOffset, hsv[2] - valOffset)
                self.dynamic_colorDict[color] = [upper, lower, id]


        #bbox_offset, cross_size = 5, 3
        bbox_offset, cross_size = 5, 3
        det_clr = []
        img_h, img_w, _ = img_orig.shape

        for i, track in enumerate(gt_tracks):
            color, colorVals = self.get_ball_color(img_orig, track, bbox_offset, cross_size, det_clr, color_only=False, use_static=True)

            if color != None:
                det_clr.append(color)

                if color not in self.detected_balls_color_value:
                    self.detected_balls_color_value[color] = [[colorVals], int(track[4].item())]
                else:
                    self.detected_balls_color_value[color][0].append(colorVals)

        update_dict()
        print('\nDynamic Dictionary Updated...')
        return


    def get_ball_color(self, img, bbox_cxcywh, bbox_offset, size, det_clr, color_only, use_static):
        cell_colors = self.__get_cell_colors(img, bbox_cxcywh, bbox_offset, size)
        # dominant_color, colorVals = self.__get_color_and_value(cell_colors, det_clr, color_only, use_static)
        return self.__get_color_and_value(cell_colors, det_clr, color_only, use_static)


    def __get_cell_colors(self, image, bbox_cxcywh, bbox_offset, size):
        def create_crosshair(image, bbox_offset, bbox_cxcywh, size):
            # For testing
            bbox = []

            # Creates a crosshair centered in the bbox with seperate areas
            cx, cy, w, h = bbox_cxcywh[0], bbox_cxcywh[1], bbox_cxcywh[2], bbox_cxcywh[3]

            # Divides bounding box into subsections
            num_splits = size * 2 + 2
            X_step = w // num_splits
            Y_step = h // num_splits

            cell_count = (size * 4) + 1
            cell_arr = np.empty((cell_count, bbox_offset * 2, bbox_offset * 2, 3), dtype=np.uint8)
            length = size * 2 + 1
            x = size * -1
            y = size * -1

            for i in range(length):
                x1 = (x * X_step) + cx
                cell_arr[i] = image[(cy - bbox_offset):(cy + bbox_offset), (x1 - bbox_offset):(x1 + bbox_offset)]

                bbox.append(((x1 - bbox_offset), (cy - bbox_offset), (x1 + bbox_offset), (cy + bbox_offset)))
                x = x + 1

            for j in range(length - 1):
                if (y == 0):
                    y = y + 1

                y1 = (y * Y_step) + cy
                ind = i + j + 1
                cell_arr[ind] = image[(y1 - bbox_offset):(y1 + bbox_offset), (cx - bbox_offset):(cx + bbox_offset)]

                bbox.append(((cx - bbox_offset), (y1 - bbox_offset), (cx + bbox_offset), (y1 + bbox_offset)))
                y = y + 1

            for i in range(cell_count):
                tmp = cell_arr[i]
                cell_arr[i] = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)

            return cell_arr

        def create_grid(image, bbox_offset, bbox_cxcywh, size):
            # For testing
            bbox = []

            # Creates a crosshair centered in the bbox with seperate areas
            cx, cy, w, h = bbox_cxcywh[0], bbox_cxcywh[1], bbox_cxcywh[2], bbox_cxcywh[3]

            # Divides bounding box into subsections
            num_splits = size * 2 + 2
            X_step = w // num_splits
            Y_step = h // num_splits

            width = size * 2 + 1
            cell_count = width * width
            x = size * -1
            y = size * -1

            # Get one pixel or an area of pixels
            if bbox_offset == -1:
                cell_arr = np.empty((cell_count, 3), dtype=np.uint8)
            else:
                cell_arr = np.empty((cell_count, bbox_offset * 2, bbox_offset * 2, 3), dtype=np.uint8)

            # Create grid
            for i in range(width):
                x1 = (x * X_step) + cx

                for j in range(width):
                    y1 = (y * Y_step) + cy
                    ind = (i * width) + j

                    if (bbox_offset == -1):
                        cell_arr[ind] = image[y1][x1]
                    else:
                        cell_arr[ind] = image[(y1 - bbox_offset):(y1 + bbox_offset),
                                       (x1 - bbox_offset):(x1 + bbox_offset)]

                    bbox.append(((x1 - bbox_offset), (y1 - bbox_offset), (x1 + bbox_offset), (y1 + bbox_offset)))
                    y = y + 1

                y = size * -1
                x = x + 1

            # Convert values to HSV
            for i in range(cell_count):
                if bbox_offset == -1:
                    cell_arr[i] = cv2.cvtColor(np.reshape(cell_arr[i], (1, 1, 3)), cv2.COLOR_BGR2HSV)
                else:
                    cell_arr[i] = cv2.cvtColor(cell_arr[i], cv2.COLOR_BGR2HSV)

            return cell_arr

        # Get HSV values from image
        if self.division_type == DivisionType.CROSSHAIR.value:
            cell_arr = create_crosshair(image, bbox_offset, bbox_cxcywh, size)
        elif self.division_type == DivisionType.GRID.value:
            cell_arr = create_grid(image, bbox_offset, bbox_cxcywh, size)

        # Return if area is 1 pixel
        if bbox_offset == -1:
            cell_colors = [(cell[0], cell[1], cell[2]) for cell in cell_arr]
            return cell_colors

        # Average each area if area is greater than 1 pixel
        arr_size = len(cell_arr)
        cell_clrs = np.empty((arr_size, 3), dtype=np.uint8)

        for i in range(arr_size):
            hue = np.mean(cell_arr[i][:, :, 0])
            sat = np.mean(cell_arr[i][:, :, 1])
            val = np.mean(cell_arr[i][:, :, 2])

            cell_clrs[i] = np.asarray((hue, sat, val))

        # Sort areas by difference from the mean
        hues = cell_clrs[:, 0]
        avg_hue = int(np.mean(hues))
        avg_diff = np.absolute(np.subtract(hues, avg_hue, dtype=np.int16))
        cell_clrs = cell_clrs[np.argsort(avg_diff, axis=0)]

        cell_colors = [(cell[0], cell[1], cell[2]) for cell in cell_clrs]

        return cell_colors


    def __get_color_and_value(self, cell_colors, det_clr, color_only, use_static):
        colorDict = self.static_colorDict if use_static else self.dynamic_colorDict
        # Count each area's color
        color_counts = dict.fromkeys(colorDict.keys(), 0)
        for color in colorDict:
            upper = colorDict[color][0]
            lower = colorDict[color][1]

            if color not in det_clr:
                for cclr in cell_colors:
                    if cclr <= upper and cclr >= lower:
                        color_counts[color] = color_counts[color] + 1

        # Find the dominant color
        dominant_color = max(color_counts, key=color_counts.get)

        if color_only:
            return dominant_color

        upper = colorDict[dominant_color][0]
        lower = colorDict[dominant_color][1]
        valid_areas = []

        for cclr in cell_colors:
            if cclr <= upper and cclr >= lower:
                valid_areas.append(cclr)

        if not valid_areas:
            return None, None

        colorVals = tuple([int(sum(clmn) / len(valid_areas)) for clmn in zip(*valid_areas)])

        return dominant_color, colorVals


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='5p2b_01A1', help='source')
    parser.add_argument('--pause_duration', type=int, default=60, help='the pause duration for catch action')
    return parser

video_resolution_dct = {
    '4p1b_01A2': (2160, 3840),
    '5p2b_01A1': (2160, 3840),
    '5p4b_01A2': (2160, 3840),
    '5p5b_03A1': (2160, 3840),
    '7p3b_02M': (1080, 1920),
}


if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    ball_reid = BallReID()

    dataset_path = os.path.join(dir_path, 'data/inputs', args.video_name)
    labels_path = os.path.join(dir_path, 'data/inputs', args.video_name, 'gt_tracks')

    video_annot_dct = load_labels_json(labels_path)
    print(sorted(video_annot_dct.keys()))

    gt_init_tracks_path = os.path.join(dir_path, 'data/inputs', args.video_name, '{}_init.csv'.format(args.video_name))

    debugging_video_path = os.path.join(dir_path, 'data/outputs', os.path.basename(args.video_name), 'calibration')

    if not os.path.exists(debugging_video_path):
        os.makedirs(debugging_video_path)

    resolution = video_resolution_dct[args.video_name]
    gt_tracks_dct = load_labels_from_csv(gt_init_tracks_path, img_h=resolution[0], img_w=resolution[1])

    vid_writer = None

    dataset = LoadImages(dataset_path)

    frame_idx = 0
    for path, img_orig, vid_cap in dataset:
        print(frame_idx)
        DrawTool.draw_frame_idx(img_orig, frame_idx, 'GroundTruth Tracks')

        repeat = 1

        if vid_writer is None:
            fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(os.path.join(debugging_video_path, '{}.m4v'.format(args.video_name)), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

        # if frame_idx in video_annot_dct:
        #     print(frame_idx)
        #     annot_bboxes = video_annot_dct[frame_idx]
        #     clses, ids, bboxes_ltrb = annot_bboxes[:, 0], annot_bboxes[:, 1], annot_bboxes[:, 2:6]
        #     DrawTool.draw_tracks(img_orig, bboxes_ltrb=bboxes_ltrb, ids=ids, clses=clses)
        #     for _ in range(repeat):
        #         vid_writer.write(img_orig)
        if frame_idx in gt_tracks_dct:
            gt_tracks = gt_tracks_dct[frame_idx]
            # t,l,w,h -> t,l,b,r
            # gt_tracks[:, 4:6] = gt_tracks[:, 2:4] + gt_tracks[:, 4:6]
            gt_clses, gt_ids, gt_bboxes_ltwh = gt_tracks[:, 0], gt_tracks[:, 1], gt_tracks[:, 2:6]
            gt_bboxes_cxcywh = np.copy(gt_bboxes_ltwh)
            gt_bboxes_cxcywh[:, :2] = gt_bboxes_cxcywh[:, :2] + 0.5 * gt_bboxes_cxcywh[:, 2:4]
            gt_tracks = np.concatenate((gt_bboxes_ltwh[gt_clses == 1], gt_ids[gt_clses == 1][:, np.newaxis]), axis=1)
            ball_reid.updateDynColorDict(gt_tracks, img_orig)

            gt_bboxes_ltrb = np.copy(gt_bboxes_ltwh)
            gt_bboxes_ltrb[:, 2:4] = gt_tracks[:, :2] + gt_tracks[:, 2:4]

            DrawTool.draw_tracks(img_orig, bboxes_ltrb=gt_bboxes_ltrb, ids=gt_ids, clses=gt_clses)
            for _ in range(repeat):
                vid_writer.write(img_orig)
        frame_idx += 1