import os
import cv2
import json
import torch
import numpy as np
from torch.utils import data
import random
from pycocotools.coco import COCO
import skimage.io as io
from utils import *
import torchvision.transforms.functional as tr_F


class FewShotInstData(data.Dataset):
    def __init__(self, coco_dir, subset, folds, n_shots, roi_size=224, p_num=256):
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

        self.inst_ids = list()
        for cat_id in active_cat_ids:
            self.inst_ids += self.coco.getAnnIds(catIds=cat_id, iscrowd=False)
        if 'val' in subset:
            self.inst_ids = self.inst_ids[0:1000]

        self.n_shots = n_shots
        self.roi_size = roi_size
        self.p_num = p_num

        # get the initial contour (same for all, in [0,1])
        unit_circle = np.zeros(shape=(self.p_num, 2), dtype=np.float32)
        for i in range(self.p_num):
            thera = 1.0 * i / self.p_num * 2 * np.pi
            x = np.cos(thera)
            y = -np.sin(thera)
            unit_circle[i, 0] = x
            unit_circle[i, 1] = y
        self.init_ctr = (0.7 * unit_circle + 1) / 2

    def __len__(self):
        return len(self.inst_ids)

    def getInstByID(self, inst_id):
        # inst_id = 2193389
        inst_anno = self.coco.loadAnns(inst_id)
        inst_cat_id = inst_anno[0]['category_id']
        inst_cat = self.coco.loadCats([inst_cat_id])
        img_mask = self.coco.annToMask(inst_anno[0])

        # get the full image
        img_id = inst_anno[0]['image_id']
        img_dict = self.coco.loadImgs(img_id)[0]
        img = io.imread(img_dict['coco_url'])
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
        roi_img = cv2.resize(roi_img, (self.roi_size, self.roi_size))
        roi_img = torch.Tensor(roi_img.transpose(2, 0, 1))
        roi_img = tr_F.normalize(roi_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # preprocess the roi mask
        roi_mask = img_mask[y_min:y_max, x_min:x_max].copy()
        roi_mask = cv2.resize(roi_mask, (self.roi_size, self.roi_size))

        # get the ground truth contour (largest component only, sorted, sampled)
        all_ctrs, hierarchy = cv2.findContours(roi_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in all_ctrs]
        if areas == []:
            print('Error:***************inst_id:', inst_id)
        gt_poly = all_ctrs[np.argmax(areas)].squeeze()
        gt_ctr = UniformSampleCtr(gt_poly, self.p_num)
        gt_ctr = gt_ctr.astype(np.float) / self.roi_size

        # Visualization
        # cv2.imshow('img', img)
        # img_mask[img_mask == 1] = 255
        # cv2.imshow('img_mask', img_mask)
        # cv2.imshow('roi_img', roi_img)
        # roi_mask[roi_mask == 1] = 255
        # cv2.imshow('roi_mask', roi_mask)
        # tmp_img = np.zeros(roi_mask.shape, dtype=np.uint8)
        # tmp_ctr = (gt_ctr*self.roi_size).astype(np.int)
        # for i in range(0, len(tmp_ctr)-2):
        #     cv2.line(tmp_img, tuple(tmp_ctr[i]), tuple(tmp_ctr[i+1]), 255, 1)
        # cv2.line(tmp_img, tuple(tmp_ctr[-1]), tuple(tmp_ctr[0]), 255, 1)
        # cv2.imshow('tmp_img', tmp_img)
        # cv2.waitKey()

        return {
            "inst_id": inst_id,
            "inst_cat_id": inst_cat_id,
            "roi_img": roi_img,
            "roi_mask": torch.Tensor(roi_mask),
            "init_ctr": self.init_ctr,
            "gt_ctr": gt_ctr
        }

    def __getitem__(self, idx):
        query_inst_id = self.inst_ids[idx]
        query_inst = self.getInstByID(query_inst_id)

        same_cat_inst_ids = self.coco.getAnnIds(catIds=query_inst["inst_cat_id"], iscrowd=False)
        support_inst_ids = random.choices(same_cat_inst_ids, k=self.n_shots)
        support_insts = [self.getInstByID(inst_id) for inst_id in support_inst_ids]

        return query_inst, support_insts
