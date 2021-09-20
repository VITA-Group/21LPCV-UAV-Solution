#!/usr/bin/env python3

from scoring import DataSet, Compare
import sys
import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--submitted', default=None, help="You Submission")

	parser.add_argument('-t', '--threshold', help='Limit Cutoff of Number of Frames You Can Be Off', default=10)
	args = parser.parse_args()

	vname_csv_mapping = {
		'7p3b_02M': '7p3b_02M_action_noisy.csv',
		'5p5b_03A1': '5p5b_03A1_action_noisy.csv',
		'5p4b_01A2': '5p4b_01A2_action_noisy.csv',
		'5p2b_01A1': '5p2b_01A1_action_noisy.csv',
		'4p1b_01A2': '4p1b_01A2_action_noisy.csv'
	}

	correct = DataSet(file_name=os.path.join('data/action_labels', vname_csv_mapping[args.submitted]))
	baseline_submitted = DataSet(file_name=os.path.join('data/outputs', '{}_baseline_out.csv'.format(args.submitted)))
	improved_submitted = DataSet(file_name=os.path.join('data/outputs', '{}_improved_out.csv'.format(args.submitted)))
	print(args.submitted)
	print('Baseline Action Detection')
	print(Compare(correct, baseline_submitted, args.threshold))
	print(args.submitted)
	print('Improved Action Detection')
	print(Compare(correct, improved_submitted, args.threshold))