"""
Find out undesired coco instances and save the ID list to a json file.
"""

import os
import cv2
import json
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
import torchvision.transforms.functional as tr_F
from multiprocessing import Pool
from utils import UniformSampleCtr
from tqdm import tqdm

coco_dir = '/WD1/few-shot/ms-coco'
subset = 'train2017'
anno_file = '{}/annotations/instances_{}.json'.format(coco_dir, subset)
coco = COCO(anno_file)
all_cat_ids = coco.getCatIds()
all_inst_ids = list()
for cat_id in all_cat_ids:
    all_inst_ids += coco.getAnnIds(catIds=cat_id)
# all_inst_ids = all_inst_ids[100:1000]

bad_inst_json = '/WD1/few-shot/kang-FSS-1000/bad_inst_list.json'
if os.path.exists(bad_inst_json):
    bad_inst_list = json.load(open(bad_inst_json))
else:
    bad_inst_list = list()


def preprocess(idx):
    roi_size = 224
    p_num = 256
    inst_id = all_inst_ids[idx]
    inst_anno = coco.loadAnns(inst_id)
    inst_cat_id = inst_anno[0]['category_id']
    inst_cat = coco.loadCats([inst_cat_id])
    img_mask = coco.annToMask(inst_anno[0])

    # get the full image
    img_id = inst_anno[0]['image_id']
    img_dict = coco.loadImgs(img_id)[0]
    img = io.imread(os.path.join(coco_dir, subset, img_dict['file_name']))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # get the instance bounding box
    bbox = inst_anno[0]['bbox']
    [x_min, y_min, w, h] = inst_anno[0]['bbox']
    [x_min, y_min, w, h] = np.round(bbox).astype(np.int)

    # pad the bounding box of the instance a little bit
    pad_scale = 0.1
    x_min = x_min - w*pad_scale
    x_max = x_min + w + 2*w*pad_scale
    x_min = max(0, np.round(x_min).astype(np.int))
    x_max = min(img.shape[1] - 1, np.round(x_max).astype(np.int))
    y_min = y_min - h*pad_scale
    y_max = y_min + h + 2*h*pad_scale
    y_min = max(0, np.round(y_min).astype(np.int))
    y_max = min(img.shape[0] - 1, np.round(y_max).astype(np.int))

    # preprocess the roi image
    roi_img = img[y_min:y_max, x_min:x_max].copy()
    inst_shape = roi_img.shape
    if inst_shape[0]*inst_shape[1] < 2500:
        # cv2.imshow('img', img)
        # img_mask[img_mask == 1] = 255
        # cv2.imshow('img_mask', img_mask)
        # cv2.waitKey()
        # bad_inst_list[inst_id] = 'Too small'
        # json.dump(bad_inst_list, open(bad_inst_json, "w"), indent=4)
        bad_inst_list.append(inst_id)
        # print(idx, 'Too small')
        return
    roi_img = cv2.resize(roi_img, (roi_size, roi_size))
    roi_img = tr_F.to_tensor(roi_img)
    roi_img = tr_F.normalize(roi_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # preprocess the roi mask
    roi_mask = img_mask[y_min:y_max, x_min:x_max].copy()
    roi_mask = cv2.resize(roi_mask, (roi_size, roi_size))

    # get the ground truth contour (largest component only, sorted, sampled)
    all_ctrs, hierarchy = cv2.findContours(roi_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in all_ctrs]
    if areas == []:
        # bad_inst_list[inst_id] = 'All black'
        # json.dump(bad_inst_list, open(bad_inst_json, "w"), indent=4)
        bad_inst_list.append(inst_id)
        # print(idx, 'All black')
        return
    gt_poly = all_ctrs[np.argmax(areas)].squeeze()
    gt_ctr = UniformSampleCtr(gt_poly, p_num)
    gt_ctr = gt_ctr.astype(np.float) / roi_size
    return


if __name__ == '__main__':
    # try:
    #     pool = Pool(processes=2)
    #     pool.map(preprocess, range(len(all_inst_ids)))
    #     pool.close()
    #     pool.join()
    # finally:
    #     print(bad_inst_list)
    #     json.dump({'bad_insts': open(bad_inst_json, "w")}, out_file, indent=4)
    #     pass

    for idx in tqdm(range(len(all_inst_ids))):
        try:
            preprocess(idx)
        except:
            json.dump(bad_inst_list, open(bad_inst_json, "w"))
            print("Error", idx)
            exit()
    json.dump(bad_inst_list, open(bad_inst_json, "w"))
