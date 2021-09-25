import os
import sys
import json
import numpy as np
from shutil import copyfile
sys.path.append("/media/pi/ssd1/cocostuffapi/PythonAPI")
from pycocotools.coco import COCO


# step up coco directory
P = "/media/pi/ssd1/coco"
IMAGE_P = os.path.join(P, "images")
dataDir='/media/pi/ssd1/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
directory = 'selected_imgs'

if not os.path.exists(directory):
    os.makedirs(directory)
directory_label = 'selected_label'
if not os.path.exists(directory_label):
    os.makedirs(directory_label)

# this part is for select person
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds);

cat_ids = coco.getCatIds()
img_cnt = 0


food_key = coco.getCatIds(supNms='food')
sports_key = coco.getCatIds(supNms='sports')
animal_key = coco.getCatIds(supNms='animal')



for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]

    # person imgs size range that will be selected
    area_up = img['height'] * img['width'] / 8 / 7
    area_low = img['height'] * img['width'] / 30 / 30
    anns_ids = coco.getAnnIds(imgId, catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    person_instances = [ann for ann in anns if ann['category_id'] == 1]
    food_instances = [ann for ann in anns if ann['category_id'] in food_key]
    sports_instances = [ann for ann in anns if ann['category_id'] in sports_key]
    animal_instances = [ann for ann in anns if ann['category_id'] in animal_key]
    ball_instances = [ann for ann in anns if ann['category_id'] == 37]

    # filter out unwanted instance
    if len(food_instances) >= 1:
        continue

    if len(ball_instances) != 0:
        continue

    if len(animal_instances) >= 1:
        continue

    if len(sports_instances) == 0:
        continue

    if len(person_instances) != 1:
        continue

    # person area check
    person_areas = np.array([ann['area'] for ann in anns if ann['category_id'] == 1])
    cond = np.where(np.logical_and(person_areas >= area_low, person_areas <= area_up))[0]
    if len(cond) != len(person_areas):
        continue
    img_cnt += 1

    # for image
    src = IMAGE_P + '/train2017/' + img['file_name']
    dst = os.path.join(directory, img['file_name'])
    copyfile(src, dst)

    # for label
    output_file_name = directory_label + '/' + img['file_name'][:-4] + '.json'
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, 'w') as f:
        json.dump(person_instances, f)
    f.close()

print("selected: ", img_cnt)






