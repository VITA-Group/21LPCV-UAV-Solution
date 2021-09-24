import os
import sys
import json
import numpy as np
import cv2
import imutils
from pycocotools.coco import COCO
from skimage import io
from os import listdir
from os.path import isfile, join
import random
from random import randrange
from PIL import Image

# append coco API
sys.path.append("/media/pi/ssd1/cocostuffapi/PythonAPI")

class Label_Writer:
    def __init__(self, saving_loc):
        self.saving_loc = saving_loc

    def write_label(self, label_class, x_c, y_c, w, h):
        with open(self.saving_loc, 'a') as f:
            f.write(str(label_class) + " " + str(x_c) + " " + str(y_c) +
                    " " + str(w) + " " + str(h) + " " + "\n")


def generate_ball(mask, bbox, ratio=1 / 5):
    """
    generate ball's radius and center y,x coordinate based on person bbox and mask
    """
    ratio = random.uniform(ratio * 0.75, ratio * 1.25)
    radius = int(bbox[3] * ratio)
    ball_y = randrange(int(bbox[1]) + radius, int(bbox[1]) + int(bbox[3]) - radius)
    mask_yth_row = mask[ball_y, :]
    # get start pos and end pos
    mask_yth_idxs = np.nonzero(mask_yth_row)[0]
    try:
        ball_x = randrange(mask_yth_idxs.min(), mask_yth_idxs.max())
    except ValueError:
        return None, None, None, True
    ball_y = int(ball_y)
    ball_x = int(ball_x)
    return radius, ball_y, ball_x, False


def ball_bbox(mask, radius, ball_y, ball_x):
    """
    calculate ball's bbox w.r.t. person mask
    return as (l, t, w, h)
    """
    l = max(0, ball_x - radius)
    t = max(0, ball_y - radius)
    w = min(2 * radius, radius + mask.shape[1] - ball_x)
    w = min(w, ball_x + radius)
    h = min(2 * radius, radius + mask.shape[0] - ball_y)
    h = min(h, ball_y + radius)
    return [l, t, w, h]


def generate_ball_masked(ball, crop_coef=.98):
    """
    generated masked ball image
    :param ball:
    :return:
    """
    ball_mask = np.zeros((ball.shape[0], ball.shape[1], 3))
    center = (int(ball_mask.shape[0] / 2), int(ball_mask.shape[1] / 2))
    radius = int((ball_mask.shape[0] + ball_mask.shape[1]) / 4 * crop_coef)  # times .98 to smooth boundary
    color = (1, 1, 1)
    thickness = -1
    ball_masked = cv2.circle(ball_mask, center, radius, color, thickness) * ball
    return ball_masked


def crop_maksed_ball(l, t, w, h, radius, img_w, img_h, ball_masked):
    """
    adjust ball_masked w.r.t image before paste to image
    :param l:
    :param t:
    :param w:
    :param h:
    :param radius:
    :param img_w:
    :param img_h:
    :param ball_masked:
    :return:
    """
    ball_l, ball_t, ball_w, ball_h = 0, 0, radius * 2, radius * 2
    if w < 2 * radius or h < 2 * radius:
        if l == 0:
            ball_l = 2 * radius - w
        if l + 2 * radius >= img_w:
            ball_w = w
        if t == 0:
            ball_t = 2 * radius - h
        if t + 2 * radius >= img_h:
            ball_h = h
    ball_pasted = ball_masked[ball_t:ball_h, ball_l:ball_w, :]
    return ball_pasted

# setup coco directory
dataDir = '/media/pi/ssd1/coco'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
P = "/media/pi/ssd1/coco"
IMAGE_P = os.path.join(P, "images")

# Selected coco images and annotation directory
imgs_dir = '/media/pi/ssd1/cocostuffapi/PythonAPI/selected_v2/'
labels_dir = '/media/pi/ssd1/cocostuffapi/PythonAPI/selected_label_v2/'
onlyfiles = [f for f in listdir(labels_dir) if isfile(join(labels_dir, f))]

# output directory
imgs_syn_dir = 'imgs_syn_v2'
label_syn_dir = 'labels_syn_v2'

# get balls dirs
balls_dir = []
mypath_balls = '/media/pi/ssd1/cocostuffapi/PythonAPI/balls/balls'

crop_coef = 0.98

# creating saving dir
if not os.path.exists(imgs_syn_dir):
    os.makedirs(imgs_syn_dir)

if not os.path.exists(label_syn_dir):
    os.makedirs(label_syn_dir)


for path, subdirs, files in os.walk(mypath_balls):
    for name in files:
        balls_dir.append(os.path.join(path, name))
balls = [io.imread(ball_dir) / 255 for ball_dir in balls_dir]

# darken_coeffs = [0.3, 0.4, 1, 1]  # redundant 1s to control portion of darken selections

# other params
rotate_angles = [0, 45, 90, 135]

for label_file in onlyfiles:
    label_path = join(labels_dir, label_file)
    img_saving_dir = join(imgs_syn_dir, label_file[:-5] + '.jpg')
    label_saving_dir = join(label_syn_dir, label_file)
    label_writer = Label_Writer(saving_loc=label_saving_dir[:-5] + '.txt')

    syn_output_list = []

    with open(label_path) as f:
        anns = json.load(f)

    img = coco.loadImgs(anns[0]['image_id'])[0]
    I_original = io.imread(IMAGE_P + '/train2017/' + img['file_name']) / 255

    anns_img = np.zeros((img['height'], img['width']))

    for ann in anns:
        # append person annotation to output_file
        people_ann = {
            'group': 'people',
            'category': ann['id'],
            'bbox': ann['bbox']
        }

        # syn_output_list.append(people_ann)
        label_writer.write_label(label_class=0,
                                 x_c=(ann['bbox'][0] + ann['bbox'][2] / 2) / img['width'],
                                 y_c=(ann['bbox'][1] + ann['bbox'][3] / 2) / img['height'],
                                 w=ann['bbox'][2] / img['width'],
                                 h=ann['bbox'][3] / img['height'])

        # get mask for person segmentation
        mask = coco.annToMask(ann)
        mask_img = np.maximum(anns_img, coco.annToMask(ann) * ann['category_id'])
        mask_img_2d = mask_img.copy()
        mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)  # extend to 3d mask

        # calculate ball position/size info
        radius, ball_y, ball_x, error_flag = generate_ball(mask_img_2d, ann['bbox'])

        if error_flag:
            continue

        [l, t, w, h] = ball_bbox(mask_img_2d, radius, ball_y, ball_x)

        # get a ball sample
        ball_idx = randrange(0, len(balls))
        ball = balls[ball_idx]
        rotate_angle = random.choice(rotate_angles)
        ball_rotated = imutils.rotate(ball, rotate_angle)


        ball = cv2.resize(ball, (radius * 2, radius * 2))

        ball_masked = generate_ball_masked(ball)

        # remove ball area from original image
        color = (0, 0, 0)
        thickness = -1
        image = I_original.copy()
        image = cv2.circle(image, (ball_x, ball_y), int(radius * crop_coef), color, thickness)  # times .98 for smooth
        img_h, img_w = image.shape[0], image.shape[1]


        # cropped ball image if originial area is not complete also generate ball that will be pasted into image
        ball_pasted = crop_maksed_ball(l, t, w, h, radius, img_w, img_h, ball_masked)
        try:
            image[t:t + h, l:l + w, :] = image[t:t + h, l:l + w, :] + ball_pasted
        except IndexError:
            continue
        I_original = image * (1 - mask_img) + I_original * mask_img

        # add ball annotation to list
        # syn_output_list.append(ball_ann)

        label_writer.write_label(label_class=1,
                                 x_c=(l + w / 2) / img['width']
                                 , y_c=(t + h / 2) / img['height'],
                                 w=w / img['width'],
                                 h=h / img['height'])

    # output images and anno

    im = Image.fromarray((I_original * 255).astype(np.uint8))
    im.save(img_saving_dir)











