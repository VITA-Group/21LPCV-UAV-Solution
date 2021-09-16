import csv
import statistics
from utils.enums import ObjectCategory
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import os
import pickle

class BaselineActionDetector(object):
    def __init__(self, tracks_history, frame_idx_history, ball_ids, person_ids):
        self.gt_ball_ids = ball_ids
        self.gt_person_ids = person_ids
        self.tracks_history = [track.astype(int) for track in tracks_history]
        self.frame_idx_history = frame_idx_history
        self.history_collision_summary = []
        self.latest_bp_assoc_dct = {}
        self.EPSILON, self.MAX_DISTANCE = 1e-10, 1.0

        for bid in self.gt_ball_ids:
            self.latest_bp_assoc_dct[bid] = 0

    def get_frame_tracks_dct(self):
        frame_tracks_dct = OrderedDict()
        for i, frame_idx in enumerate(self.frame_idx_history):
            tracks = self.tracks_history[i]
            tracks[:, :2] = tracks[:, :2] + tracks[:, 2:4] / 2
            person_tracks = tracks[tracks[:, 5] == 0, :]
            ball_tracks = tracks[tracks[:, 5] == 1, :]
            if person_tracks.size > 0 and ball_tracks.size > 0:
                frame_tracks_dct[frame_idx] = (ball_tracks, person_tracks)
        return frame_tracks_dct

    def update_catches(self, tracks, frame_idx):
        def get_bp_collision_dist(ball_tracks, person_tracks):
            balls_center, persons_center = ball_tracks[:, :2], person_tracks[:, :2]
            persons_lt = person_tracks[:, :2] - 0.5 * person_tracks[:, 2:4]
            persons_rb = person_tracks[:, :2] + 0.5 * person_tracks[:, 2:4]
            bp_collistion_matx = np.logical_and(
                np.logical_and(*np.dsplit(
                    np.subtract(balls_center[:, np.newaxis, :], persons_lt[np.newaxis, :, :]) >= 0, 2)),
                np.logical_and(*np.dsplit(
                    np.subtract(persons_rb[np.newaxis, :, :], balls_center[:, np.newaxis, :]) >= 0, 2))
            )
            bp_dist_matx = np.linalg.norm(
                np.subtract(balls_center[:, np.newaxis, :], persons_center), axis=2)
            person_diag_matx = np.tile(np.linalg.norm(np.subtract(persons_lt, persons_rb), axis=-1),
                                       (ball_tracks.shape[0], 1))
            bp_norm_dist_matx = np.true_divide(bp_dist_matx, person_diag_matx)
            return bp_collistion_matx.reshape((ball_tracks.shape[0], person_tracks.shape[0])), \
                   bp_norm_dist_matx.reshape((ball_tracks.shape[0], person_tracks.shape[0]))

        ball_tracks, person_tracks = tracks

        bp_collistion_matx, bp_norm_dist_matx = get_bp_collision_dist(ball_tracks, person_tracks)
        cost_matrix = np.multiply(bp_norm_dist_matx+self.EPSILON, bp_collistion_matx)
        cost_matrix[cost_matrix <= self.EPSILON] = self.MAX_DISTANCE

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        collisions = []
        for row, col in zip(row_indices, col_indices):
            ball_id, person_id = ball_tracks[row, 4], person_tracks[col, 4]
            if cost_matrix[row, col] < self.MAX_DISTANCE:
                collisions.append((person_id, ball_id))

        # collision_pos = np.where(bp_collistion_matx == 1)
        # collision_balls_indx, collision_persons_indx = collision_pos[0], collision_pos[1]
        # collision_balls_id, collision_persons_id = ball_tracks[collision_balls_indx, 4], person_tracks[collision_persons_indx, 4]
        # _, indices = np.unique(collision_balls_id, return_index=True)
        # pb_collision_relations = dict(zip(collision_persons_id[indices], collision_balls_id[indices]))

        pb_collision_relations = dict(collisions)
        self.update_bp_assoc_dct(frame_idx, pb_collision_relations)

    def write_catches(self, output_path):
        frame_tracks_dct = self.get_frame_tracks_dct()
        for frame_idx, tracks in frame_tracks_dct.items():
            self.update_catches(tracks, frame_idx)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            ordered_balls = sorted(self.latest_bp_assoc_dct.keys())
            ordered_balls.insert(0, "frame")
            writer.writerow(ordered_balls)
            self.correct_catches()
            for i in range(len(self.history_collision_summary)):
                frame_idx = self.history_collision_summary[i][0]
                frame_assoc = self.history_collision_summary[i][1]
                frame_assoc.insert(0, frame_idx)
                writer.writerow(frame_assoc)
        return

    def correct_catches(self):
        num_balls = len(self.history_collision_summary[-1][1])
        window_size = 20
        size = len(self.history_collision_summary)
        frame_catch_frame_assoc = []

        i = 0
        while i < size:
            current_frame_summary = self.history_collision_summary[i]
            current_frame_idx, current_frame_assoc = current_frame_summary[0], current_frame_summary[1]

            balls_assoc_history = [[] for _ in range(num_balls)]
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

def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    vid_name = '7p3b_02M'
    dst = 'outputs'
    outpath = os.path.join(dst, vid_name, vid_name + '_out.csv')
    saved_tracks_path = os.path.join(dst, vid_name, 'tracking')

    tracks_history = load_pkl(os.path.join(saved_tracks_path, 'tracks_history.pkl'))
    frames_idx_history = load_pkl(os.path.join(saved_tracks_path, 'frames_idx_history.pkl'))
    gt_pids = load_pkl(os.path.join(saved_tracks_path, 'person_ids.pkl'))
    gt_bids = load_pkl(os.path.join(saved_tracks_path, 'ball_ids.pkl'))

    action_detector = OnlineActionDetector(tracks_history, frames_idx_history, gt_bids, gt_pids)
    action_detector.write_catches(outpath)