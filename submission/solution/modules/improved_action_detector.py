import numpy as np
import sys
from collections import OrderedDict
import csv
import math
from itertools import groupby
import os
import pickle
from scipy.optimize import linear_sum_assignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.experimental import save_pkl, load_pkl

class ImprovedActionDetector(object):
    def __init__(self, tracks_history, frame_idx_history, ball_ids, person_ids):
        self.gt_balls_id = ball_ids
        self.gt_persons_id = person_ids
        self.tracks_history = [track.astype(int) for track in tracks_history]
        self.frame_idx_history = frame_idx_history
        self.persons_count = len(self.gt_persons_id)
        self.balls_count = len(self.gt_balls_id)

        self.morph_radius = 5
        self.pred_frames = []
        self.EPSILON, self.MAX_DISTANCE = 1e-10, 1.0

    def get_frame_tracks_dct(self):
        frame_tracks_dct = OrderedDict()
        for i, frame_idx in enumerate(self.frame_idx_history):
            tracks = self.tracks_history[i]
            # tracks[:, :2] = tracks[:, :2] + tracks[:, 2:4] / 2
            person_tracks = tracks[tracks[:, 5] == 0, :]
            ball_tracks = tracks[tracks[:, 5] == 1, :]
            if person_tracks.size > 0 and ball_tracks.size > 0:
                frame_tracks_dct[frame_idx] = (ball_tracks, person_tracks)
        return frame_tracks_dct

    def get_dist_collision_history(self, frame_tracks_dct):
        def collision(ball_tracks, person_tracks):
            balls_center = 0.5 * (ball_tracks[:, :2] + ball_tracks[:, 2:4])
            persons_center = 0.5 * (person_tracks[:, :2] + person_tracks[:, 2:4])
            persons_lt = person_tracks[:, :2]
            persons_rb = person_tracks[:, 2:4]
            ball_ids, person_ids = ball_tracks[:, 4], person_tracks[:, 4]
            bp_dist_matx = np.linalg.norm(
                                    np.subtract(balls_center[:, np.newaxis, :], persons_center), axis=2).reshape((len(ball_ids), len(person_ids)))
            person_diag_matx = np.tile(np.linalg.norm(np.subtract(persons_lt, persons_rb), axis=-1),
                                       (ball_tracks.shape[0], 1))
            bp_norm_dist_matx = np.true_divide(bp_dist_matx, person_diag_matx).reshape((len(ball_ids), len(person_ids)))
            bp_collistion_matx = np.logical_and(
                                    np.logical_and(*np.dsplit(
                                        np.subtract(balls_center[:, np.newaxis, :], persons_lt[np.newaxis, :, :]) > 0, 2)),
                                    np.logical_and(*np.dsplit(
                                        np.subtract(persons_rb[np.newaxis, :, :], balls_center[:, np.newaxis, :]) > 0, 2))
                                    ).reshape((len(ball_ids), len(person_ids)))
            if np.any(bp_collistion_matx):
                return True, bp_norm_dist_matx, bp_collistion_matx, ball_ids, person_ids
            return False, None, None, None, None

        pred_frames = []
        bp_norm_dist_history_dct, bp_collision_history_dct, bp_ids_history_dct = {}, {}, {}
        for frame_idx, tracks in frame_tracks_dct.items():
            ball_tracks, person_tracks = tracks
            has_collision, bp_norm_dist_matx, bp_collistion_matx, ball_ids, person_ids = collision(ball_tracks, person_tracks)
            if has_collision:
                pred_frames.append(frame_idx)
                bp_norm_dist_history_dct[frame_idx] = bp_norm_dist_matx
                bp_collision_history_dct[frame_idx] = bp_collistion_matx
                bp_ids_history_dct[frame_idx] = (ball_ids, person_ids)
        return pred_frames, bp_norm_dist_history_dct, bp_collision_history_dct, bp_ids_history_dct

    def write_catches(self, outpath):
        frame_tracks_dct = self.get_frame_tracks_dct()
        pred_frames, bp_norm_dist_history_dct, bp_collision_history_dct, bp_ids_history_dct = self.get_dist_collision_history(frame_tracks_dct)

        bp_catch_records_dct = {}
        for id in self.gt_balls_id:
            bp_catch_records_dct[id] = np.zeros((self.persons_count, self.frame_idx_history[-1]+1,), dtype=int)

        gt_persons_id_arr = np.array(self.gt_persons_id)
        for frame_idx in pred_frames:
            # print(frame_idx)
            bp_norm_dist_matx  = bp_norm_dist_history_dct[frame_idx]
            bp_collistion_matx = bp_collision_history_dct[frame_idx]
            ball_ids, person_ids = bp_ids_history_dct[frame_idx]
            cost_matrix = np.multiply(bp_norm_dist_matx + self.EPSILON, bp_collistion_matx)
            cost_matrix[cost_matrix <= self.EPSILON] = self.MAX_DISTANCE
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            collisions = []
            for row, col in zip(row_indices, col_indices):
                ball_id, person_id = ball_ids[row], person_ids[col]
                if cost_matrix[row, col] < self.MAX_DISTANCE:
                    collisions.append((ball_id, person_id))
            # collision_pos = np.where(bp_collistion_matx==1)
            # collision_balls_indx, collision_persons_indx = collision_pos[0], collision_pos[1]
            # collision_balls_id, collision_persons_id = ball_ids[collision_balls_indx], person_ids[collision_persons_indx]
            # bp_collision_relations = dict(zip(collision_balls_id, collision_persons_id))

            bp_collision_relations = dict(collisions)
            for bid, pid in bp_collision_relations.items():
                pid_indx = np.where(gt_persons_id_arr == pid)[0].item()
                bp_catch_records_dct[bid][pid_indx, frame_idx] = 1

        catch_bp_assoc_history = {}
        for bid in self.gt_balls_id:
            time_pid_pairs = self.morph(bp_catch_records_dct[bid])
            catch_bp_assoc_history[bid] = time_pid_pairs

        gt_bids_arr = np.array(self.gt_balls_id)
        order = np.argsort(gt_bids_arr)
        ordered_balls = gt_bids_arr[order].tolist()
        outcome = self.format_writing(catch_bp_assoc_history, ordered_balls)
        with open(outpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['frame'] + ordered_balls)
            for line in outcome:
                writer.writerow(line)

    def format_writing(self, catch_bp_assoc_history, ball_order):
        records = []
        latest_bp_assoc_dct = {}
        for bid, time_pid_pairs in catch_bp_assoc_history.items():
            latest_bp_assoc_dct[bid] = time_pid_pairs[0][1]
            for time, pid in time_pid_pairs:
                records.append((time, bid, pid))

        sorted_records = sorted(records, key=lambda foo:foo[0])

        outcome = []
        for i, record in enumerate(sorted_records):
            time, bid, pid = record[0], record[1], record[2]
            latest_bp_assoc_dct[bid] = pid
            if i + 1 == len(sorted_records):
                next_time = math.inf
            else:
                next_time = sorted_records[i+1][0]
            if time < next_time:
                pids = [latest_bp_assoc_dct[b] for b in ball_order]
                outcome.append([sorted_records[i][0]] + pids )
        return outcome

    def morph(self, raw_signal):
        def find_consecutive_ones(arr):
            isOne = np.concatenate(([0], (arr > 0).astype(int), [0]))
            absdiff = np.abs(np.diff(isOne))
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            return ranges

        padded_raw_signal_mtx = np.concatenate((np.zeros((self.persons_count, self.morph_radius)),
                                            raw_signal, np.zeros((self.persons_count, self.morph_radius))), axis=1)

        padded_raw_signal_vec = np.sum(padded_raw_signal_mtx, axis=0)
        # Padding with 0s
        padded_gated_signal_vec = (padded_raw_signal_vec > 0).astype(int)
        for i in range(padded_raw_signal_vec.size):
            if padded_raw_signal_vec[i] > 0:
                left = i - self.morph_radius
                right = i + self.morph_radius + 1
                padded_gated_signal_vec[left:right] = 1
        ranges = find_consecutive_ones(padded_gated_signal_vec).tolist()

        time_pid_pairs = []
        pids = []
        for left, right in ranges:
            interval = padded_raw_signal_mtx[:, left:right]
            pid_indx = np.argmax(np.sum(interval, axis=1))
            pid = self.gt_persons_id[pid_indx]
            # valid_ids = interval[interval > 0]
            # pid = int(statistics.mode(valid_ids))
            time_pid_pairs.append((left, pid))
            pids.append(pid)

        grouped_indices = [list(g) for _, g in groupby(range(len(pids)), lambda idx: pids[idx])]
        group_ranges = [[indices[0],indices[-1]] for indices in grouped_indices]

        merged_time_pid_pair = []

        for gr in group_ranges:
            merged_time_pid_pair.append(time_pid_pairs[gr[0]])
        return merged_time_pid_pair

if __name__ == '__main__':
    vid_name = '7p3b_02M'
    dst = 'data/outputs'
    outpath = os.path.join(dst, vid_name, vid_name + '_out.csv')
    saved_tracks_path = os.path.join(dst, vid_name, 'tracking')

    tracks_history = load_pkl(os.path.join(saved_tracks_path, 'tracks_history.pkl'))
    frames_idx_history = load_pkl(os.path.join(saved_tracks_path, 'frames_idx_history.pkl'))
    gt_pids = load_pkl(os.path.join(saved_tracks_path, 'person_ids.pkl'))
    gt_bids = load_pkl(os.path.join(saved_tracks_path, 'ball_ids.pkl'))

    action_detector = ImprovedActionDetector(tracks_history, frames_idx_history, gt_bids, gt_pids)
    action_detector.write_catches(outpath)