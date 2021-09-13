import csv
import statistics

from collections import OrderedDict

from utils.enums import ObjectCategory
from ball_reid import Box

class OnlineActionDetector(object):
    def __init__(self, ball_ids, person_ids):
        self.ball_ids = ball_ids
        self.person_ids = person_ids
        self.history_collision_summary = []
        self.latest_bp_assoc_dct = OrderedDict()
        for bid in ball_ids:
            self.latest_bp_assoc_dct[bid] = 0

    def update_catches(self, tracks, frame_idx):

        bboxes_ltwh, ids, clses = tracks[:, :4], tracks[:, 4], tracks[:, 5]

        # Create a list of bbox centers and ranges
        bboxes_cxcyRange = Box.xyxy2cxcyRange_batch(bboxes_ltwh)

        persons_bbox_cxcyRange = [[ids[i]] + bbox for i, bbox in enumerate(bboxes_cxcyRange) if clses[i] == ObjectCategory.PERSON.value and ids[i] in self.person_ids]
        balls_bbox_cxcyRange = [[ids[i]] + bbox for i, bbox in enumerate(bboxes_cxcyRange) if clses[i] == ObjectCategory.BALL.value and ids[i] in self.ball_ids]

        # Detect collison between balls and people
        collisions = self.detect_bp_collisions(persons_bbox_cxcyRange, balls_bbox_cxcyRange)

        self.update_bp_assoc_dct(frame_idx, collisions)


    def write_catches(self, output_path):
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            ordered_balls = sorted(self.latest_bp_assoc_dct.keys())
            ordered_balls.insert(0, "frame")
            writer.writerow(ordered_balls)
            self.catch_action_correction()
            for i in range(len(self.history_collision_summary)):
                frame_idx = self.history_collision_summary[i][0]
                frame_assoc = self.history_collision_summary[i][1]
                frame_assoc.insert(0, frame_idx)
                writer.writerow(frame_assoc)
        return

    def detect_bp_collisions(self, persons_bbox_cxcyRange, balls_bbox_cxcyRange):

        def collision_criterion(person_xrng, person_yrng, ball_cx, ball_cy):
            if ball_cx >= person_xrng[0] and ball_cx <= person_xrng[1] \
                    and ball_cy >= person_yrng[0] and ball_cy <= person_yrng[1]:
                return True
            else:
                return False

        collisions = {} # key: id, value: color

        for pbox in persons_bbox_cxcyRange:
            person_id, person_xrng, person_yrng = pbox[0], pbox[3], pbox[4]
            for bbox in balls_bbox_cxcyRange:
                ball_id, ball_cx, ball_cy = bbox[0], bbox[1], bbox[2]
                if collision_criterion(person_xrng, person_yrng, ball_cx, ball_cy) and ball_id not in collisions.values():
                    collisions[person_id] = ball_id
                    break
        return collisions

    def update_bp_assoc_dct(self, frame_idx, collisions):
        updateCatchAction = False

        for person in collisions:
            ball = collisions[person]
            # Ball has not been caught yet
            if ball not in self.latest_bp_assoc_dct:
                self.latest_bp_assoc_dct[ball] = person

            # Ball is caught by a new person
            elif self.latest_bp_assoc_dct[ball] != person:
                self.latest_bp_assoc_dct[ball] = person
                updateCatchAction = True

        if updateCatchAction:
            ordered_balls = sorted(self.latest_bp_assoc_dct.keys())
            summary = [self.latest_bp_assoc_dct[ball_id] for ball_id in ordered_balls]
            self.history_collision_summary.append([frame_idx, summary])
        return

    def catch_action_correction(self):
        num_balls = len(self.history_collision_summary[-1][1])
        window_size = 20
        size = len(self.history_collision_summary)
        frame_catch_frame_assoc = []

        i = 0
        while i < size:
            current_frame_summary = self.history_collision_summary[i]
            current_frame_idx, current_frame_assoc = current_frame_summary[0], current_frame_summary[1]

            balls_assoc_history = [[] for _ in range(num_balls)]
            print(i)
            for k, id in enumerate(current_frame_assoc):
                balls_assoc_history[k].append(id)
            j = i + 1

            while j < size:
                next_frame_summary = self.history_collision_summary[j]
                next_frame_idx, next_frame_assoc = next_frame_summary[0], next_frame_summary[1]
                idx_diff = next_frame_idx - current_frame_idx
                if idx_diff < window_size:
                    for k, id in enumerate(next_frame_assoc):
                        balls_assoc_history[k].append(id)
                    j = j + 1
                else:
                    break

            assoc_summary = [int(statistics.mode(ball_assoc)) for ball_assoc in balls_assoc_history]
            frame_catch_frame_assoc.append([current_frame_idx, assoc_summary])

            i = j

        self.history_collision_summary = frame_catch_frame_assoc
        return