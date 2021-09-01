import numpy as np
import statistics

from numpy import linalg as la
from collections import OrderedDict
import csv
import math
from itertools import groupby
from collections import defaultdict
import os
from utils.experimental import (save_pkl, load_pkl)

class OfflineActionDetector(object):
    def __init__(self, tracks_history, frame_idx_history, ball_ids, person_ids):
        self.gt_balls_id = ball_ids
        self.gt_persons_id = person_ids
        self.tracks_history = [track.astype(int) for track in tracks_history]
        self.frame_idx_history = frame_idx_history
        self.persons_count = len(self.gt_persons_id)
        self.balls_count = len(self.gt_balls_id)

        self.morph_radius = 5
        self.key_frames = []

    def write_catches(self, outpath):
        frame_tracks_dct = self.get_frame_tracks_dct()
        key_frames, bp_dist_history_dct, bp_collision_history_dct, bp_ids_history_dct = self.get_dist_collision_history(frame_tracks_dct)

        bp_catch_records_dct = {}
        for id in self.gt_balls_id:
            bp_catch_records_dct[id] = np.zeros((self.persons_count, self.frame_idx_history[-1]+1,), dtype=int)

        gt_persons_id_arr = np.array(self.gt_persons_id)
        for frame_idx in key_frames:
            print(frame_idx)
            # bp_dist_matx  = bp_dist_history_dct[frame_idx]
            bp_collistion_matx = bp_collision_history_dct[frame_idx]
            ball_ids, person_ids = bp_ids_history_dct[frame_idx]
            collision_pos = np.where(bp_collistion_matx==1)
            collision_balls_indx, collision_persons_indx = collision_pos[0], collision_pos[1]
            collision_balls_id, collision_persons_id = ball_ids[collision_balls_indx], person_ids[collision_persons_indx]
            bp_collision_relations = dict(zip(collision_balls_id, collision_persons_id))
            for bid, pid in bp_collision_relations.items():
                pid_indx = np.where(gt_persons_id_arr == pid)[0].item()
                bp_catch_records_dct[bid][pid_indx, frame_idx] = 1
            # collision_rindices = np.any(bp_collistion_matx, axis=1)
            # bp_dist_matx_collision = bp_dist_matx[collision_rindices, :]
            # nn_cindices = np.argmin(bp_dist_matx_collision, axis=1)
            # active_balls, active_persons = ball_ids[collision_rindices], person_ids[nn_cindices]
            # for i, ball in enumerate(active_balls):
            #     bp_catch_records_dct[ball][frame_idx] = active_persons[i]
        # print(bp_catch_records_dct)
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

    def get_dist_collision_history(self, frame_tracks_dct):
        def collision(ball_tracks, person_tracks):
            balls_center, persons_center = ball_tracks[:, :2], person_tracks[:, :2]
            ball_ids, person_ids = ball_tracks[:, 4], person_tracks[:, 4]
            bp_dist_matx = np.linalg.norm(
                                    np.subtract(balls_center[:, np.newaxis, :], persons_center), axis=2).reshape((len(ball_ids), len(person_ids)))
            persons_lt = person_tracks[:, :2] - 0.5 * person_tracks[:, 2:4]
            persons_rb = person_tracks[:, :2] + 0.5 * person_tracks[:, 2:4]
            bp_collistion_matx = np.logical_and(
                                    np.logical_and(*np.dsplit(
                                        np.subtract(balls_center[:, np.newaxis, :], persons_lt[np.newaxis, :, :]) > 0, 2)),
                                    np.logical_and(*np.dsplit(
                                        np.subtract(persons_rb[np.newaxis, :, :], balls_center[:, np.newaxis, :]) > 0, 2))
                                    ).reshape((len(ball_ids), len(person_ids)))
            if np.any(bp_collistion_matx):
                return True, bp_dist_matx, bp_collistion_matx, ball_ids, person_ids
            return False, None, None, None, None

        key_frames = []
        bp_dist_history_dct, bp_collision_history_dct, bp_ids_history_dct = {}, {}, {}
        for frame_idx, tracks in frame_tracks_dct.items():
            ball_tracks, person_tracks = tracks
            has_collision, bp_dist_matx, bp_collistion_matx, ball_ids, person_ids = collision(ball_tracks, person_tracks)
            if has_collision:
                key_frames.append(frame_idx)
                bp_dist_history_dct[frame_idx] = bp_dist_matx
                bp_collision_history_dct[frame_idx] = bp_collistion_matx
                bp_ids_history_dct[frame_idx] = (ball_ids, person_ids)
        return key_frames, bp_dist_history_dct, bp_collision_history_dct, bp_ids_history_dct

import pickle

def save_dict(dct, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dct, f)

def load_dict(filename):
    with open(filename, 'rb') as f:
        dct = pickle.load(f)
    return dct


if __name__ == '__main__':
    vid_name = '7p3b_02M'
    dst = 'outputs'
    outpath = os.path.join(dst, vid_name, vid_name + '_out.csv')
    saved_tracks_path = os.path.join(dst, vid_name, 'tracking')

    tracks_history = load_pkl(os.path.join(saved_tracks_path, 'tracks_history.pkl'))
    frames_idx_history = load_pkl(os.path.join(saved_tracks_path, 'frames_idx_history.pkl'))
    gt_tracks = load_pkl(os.path.join(saved_tracks_path, 'gt_tracks.pkl'))

    action_detector = OfflineActionDetector(tracks_history, gt_tracks, frames_idx_history)
    action_detector.write_catches(outpath)