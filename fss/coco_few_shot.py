import os
import cv2
import json
import torch
import numpy as np
from torch.utils import data
import random
from pycocotools.coco import COCO
import skimage.io as io


class FewShotInstData(data.Dataset):
    def __init__(self, coco_dir, subset, folds, n_shots, img_size=224, exclude_list_file=None, resize_flag=True):
        '''
        :param coco_dir: The path of COCO folder.
        :param subset: 'train2017', 'val2017' or 'test2017'.
        :param folds: Split the subset into 4 folds for cross-validation. Any in [0,1,2,3].
        :param n_shots: Number of support images.
        '''
        anno_file = '{}/annotations/instances_{}.json'.format(coco_dir, subset)
        self.coco = COCO(anno_file)
        all_cat_ids = self.coco.getCatIds()

        active_cat_ids = list()
        for fold in folds:
            active_cat_ids += all_cat_ids[fold*20:(fold+1)*20]

        if 'train' in subset and exclude_list_file is not None:
            excluded_inst_ids = json.load(open(exclude_list_file))
        else:
            excluded_inst_ids = []

        self.inst_ids = list()
        self.cat_inst_ids = {}
        for cat_id in active_cat_ids:
            self.cat_inst_ids[cat_id] = self.coco.getAnnIds(catIds=cat_id, iscrowd=False)
            if excluded_inst_ids:
                self.cat_inst_ids[cat_id] = list(set(self.cat_inst_ids[cat_id]) - set(excluded_inst_ids))
            self.inst_ids += self.cat_inst_ids[cat_id]
        print('Total inst. #:', len(self.inst_ids))
        # cat_insts_cnt = {cat_id: len(cat_inst_ids) for cat_id, cat_inst_ids in self.cat_inst_ids.items()}
        # print(cat_insts_cnt)

        self.n_shots = n_shots
        self.img_size = img_size
        self.coco_dir = coco_dir
        self.subset = subset
        self.resize_flag = resize_flag

    def __len__(self):
        return len(self.inst_ids)

    def getInstByID(self, inst_id):
        inst_anno = self.coco.loadAnns(inst_id)
        inst_cat_id = inst_anno[0]['category_id']
        inst_cat = self.coco.loadCats([inst_cat_id])
        img_mask = self.coco.annToMask(inst_anno[0])

        img_id = inst_anno[0]['image_id']
        img_dict = self.coco.loadImgs(img_id)[0]
        # img = io.imread(os.path.join(self.coco_dir, self.subset, img_dict['file_name']))
        img = cv2.imread(os.path.join(self.coco_dir, self.subset, img_dict['file_name']))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

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
        roi = img[y_min:y_max, x_min:x_max].copy()
        roi_mask = img_mask[y_min:y_max, x_min:x_max].copy()

        if self.resize_flag:
            try:
                roi = cv2.resize(roi, (self.img_size, self.img_size))
                roi_mask = cv2.resize(roi_mask, (self.img_size, self.img_size))
            except:
                print(img_id, inst_id, img_dict['file_name'], inst_cat_id)
                print('Orig:', x_min, y_min, x_max, y_max, w, h)
                raise ValueError('ROI error')

        inst = {
            "inst_id": inst_id,
            "inst_cat_id": inst_cat_id,
            "roi": roi,
            "roi_mask": roi_mask
        }

        return inst

    def __getitem__(self, idx):
        query_inst_id = self.inst_ids[idx]
        query_inst = self.getInstByID(query_inst_id)

        same_cat_inst_ids = self.cat_inst_ids[query_inst["inst_cat_id"]].copy()
        same_cat_inst_ids.remove(query_inst_id)
        support_inst_ids = random.choices(same_cat_inst_ids, k=self.n_shots)
        support_insts = [self.getInstByID(inst_id) for inst_id in support_inst_ids]

        return query_inst, support_insts
