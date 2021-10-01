import csv
import statistics


from enums import ObjectCategory, DivisionType, BallStatus
from ball_reid import Ball

class ActionDetector(object):
	def __init__(self):
		self.frame_catch_pairs = []
		self.ball_person_association_dct = {}

	def update_catches(self, image, bboxes_xyxy, classes, ids, frame_idx, ball_reid):
		# Create a list of bbox centers and ranges
		bboxes_cxcyRange = Ball.xyxy2cxcyRange_batch(bboxes_xyxy)

		# Detect the color of each ball and return a dictionary matching id to color
		detected_ball_colors = self.__detect_balls_color(image, bboxes_cxcyRange, classes, ids, ball_reid)

		# Detect collison between balls and people
		collisions = self.__detect_collisions(classes, ids, bboxes_cxcyRange, detected_ball_colors)

		# Update dictionary pairs
		self.__update_dict_pairs(frame_idx, collisions, list(ball_reid.dynamic_colorDict.keys()))
		balls_status_str = self.__format_balls_status(ids, classes, detected_ball_colors, collisions)

		return balls_status_str


	def write_catches(self, output_path, ball_reid):
		colorDict = ball_reid.dynamic_colorDict
		colorOrder = list(ball_reid.dynamic_colorDict.keys())
		with open(output_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			ball_ids = []
			for color in colorOrder:
				ball_ids.append(colorDict[color][2])
			ball_ids.insert(0, "frame")
			writer.writerow(ball_ids)
			self.__smooth_frame_pairs(ball_reid)
			for i in range(len(self.frame_catch_pairs)):
				frame = self.frame_catch_pairs[i][0]
				pairs = self.frame_catch_pairs[i][1].split(' ')
				pairs.insert(0, frame)
				writer.writerow(pairs)
		return


	def __detect_balls_color(self, image, bboxes_cxcyRange, classes, ids, ball_reid):
		# Cross_size is the number of radial members around the center
		# Bbox_offset is the radius of the bbox
		detected_balls_color_cxcy = {}
		det_clr = []
		bbox_offset = -1
		size = 4

		for i in range(len(classes)):

			# Checks if the class is a ball (1)
			if classes[i] == ObjectCategory.BALL.value:
				ball_color = ball_reid.get_ball_color(image, bboxes_cxcyRange[i], bbox_offset, size,
				                                      det_clr, color_only=True, use_static=False)

				det_clr.append(ball_color)
				detected_balls_color_cxcy[ids[i]] = [ball_color, bboxes_cxcyRange[i][0], bboxes_cxcyRange[i][1]]

		return detected_balls_color_cxcy


	def __detect_collisions(self, classes, ids, bboxes_cxcyRange, detected_balls_color_cxcy):
		collisions = {} # key: id, value: color

		maxId = 10 # ID larger than maxID is very likely to be wrong tracking

		for i in range(len(classes)):
			# Check if the track is a person
			if classes[i] == ObjectCategory.PERSON.value and ids[i] < maxId:

				# Get persons bbox range
				person_X_range, person_Y_range = bboxes_cxcyRange[i][2], bboxes_cxcyRange[i][3]

				# Check if the center of a ball is in a persons bounding box
				# detected_balls_color_cxcy = {'id' : [color, cx, cy], ...}
				for ball in detected_balls_color_cxcy:
					ball_color, ball_cx, ball_cy = detected_balls_color_cxcy[ball][0], detected_balls_color_cxcy[ball][1], detected_balls_color_cxcy[ball][2]

					if ball_cx >= person_X_range[0] and ball_cx <= person_X_range[1] and ball_cy >= person_Y_range[0] and ball_cy <= person_Y_range[1] and ball_color not in collisions.values():
						collisions[ids[i]] = ball_color
						break

		return collisions


	def __update_dict_pairs(self, frame_idx, collisions, colorOrder):
		updateCatchAction = False

		for person in collisions:
			ball = collisions[person]
			# Ball color has not been held yet
			if ball not in self.ball_person_association_dct:
				self.ball_person_association_dct[ball] = person

			# Ball is held by a new person
			elif self.ball_person_association_dct[ball] != person:
				# for ball in self.ball_person_association_dct:
				# 	if self.ball_person_association_dct[ball] == person:
				# 		self.ball_person_association_dct[ball] = 0
				self.ball_person_association_dct[ball] = person
				updateCatchAction = True

		if updateCatchAction:
			summaryStr = ''
			for color in colorOrder:
				if color in self.ball_person_association_dct:
					summaryStr = summaryStr + str(self.ball_person_association_dct[color]) + ' '
				else:
					summaryStr = summaryStr + '0' + ' '
			self.frame_catch_pairs.append([frame_idx, summaryStr])
		return


	def __format_balls_status(self, ids, classes, detected_ball_colors, collisions):
		bbox_strings = [None] * len(classes)

		for i in range(len(classes)):
			# Person bbox info
			if ids[i] in collisions:
				color = collisions[ids[i]]
				txt = '{status} {color}'.format(status=BallStatus.CAUGHT.name, color=color)

			# Ball bbox info
			elif ids[i] in detected_ball_colors:
				color = detected_ball_colors[ids[i]][0]
				txt = '{status} {color}'.format(status=BallStatus.FLYING.name, color=color)

			else:
				txt = ''

			bbox_strings[i] = txt

		return bbox_strings


	def __smooth_frame_pairs(self, ball_reid):
		max_diff = 5
		size = len(self.frame_catch_pairs)
		num_clrs = len(ball_reid.dynamic_colorDict)
		smooth_pairs = []

		i = 0
		while i < size:
			frame = self.frame_catch_pairs[i][0]

			# Check if next item is in range
			if ((i + 1) < size):
				diff = self.frame_catch_pairs[i + 1][0] - frame

				# Check if next frame is close
				if (diff < max_diff):
					color_ids = [None] * num_clrs
					# color_ids = [[], [], [], [], [], []]
					# for k in range(num_clrs):
					# 	color_ids.append([])
					tmp_frames = self.frame_catch_pairs[i:]
					nxt_i = i

					for cur_frame in tmp_frames:
						cur_ids = cur_frame[1][:-1]
						cur_ids = cur_ids.split(' ')
						cur_dif = cur_frame[0] - frame

						if cur_dif < max_diff:
							for k in range(len(cur_ids)):
								color_ids[k].append(cur_ids[k])
							nxt_i = nxt_i + 1
						else:
							break

					tmp = ''
					for j in range(len(color_ids)):
						mode = statistics.mode(color_ids[j])
						tmp = tmp + mode + ' '

					i = nxt_i
					smooth_pairs.append([frame, tmp])
				else:
					smooth_pairs.append(self.frame_catch_pairs[i])
					i = i + 1

			else:
				smooth_pairs.append(self.frame_catch_pairs[i])
				i = i + 1
		self.frame_catch_pairs = smooth_pairs
		return