import os
import cv2
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
from dataloaders.coco_few_shot import FewShotInstData
from networks.few_shot_net import FewShotNet
import pdb


def test(opts):
    # Prepare data
    test_dataset = FewShotInstData(opts, subset='train2017', folds=[3])
    test_loader = DataLoader(test_dataset, batch_size=opts['batch_size'], num_workers=opts['num_workers'], shuffle=False, pin_memory=True)

    model = FewShotNet(opts).cuda()
    model = nn.DataParallel(model, device_ids=opts['gpu_list'])
    model.load_state_dict(torch.load(os.path.join(opts['ckp_dir'], 'best.pth')))
    model.eval()

    iou = 0.0
    intersect_sum, union_sum = 0.0, 0.0
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            qry_inst, supp_insts = batch_data

            PA_qry_pred, PA_align_loss = model(qry_inst, supp_insts)
            PA_qry_mask = np.array(PA_qry_pred.argmax(dim=1).cpu()).astype(np.uint8)
            gt_mask = np.array(qry_inst['roi_mask']).astype(np.uint8)

            # For visualization
            for i in range(gt_mask.shape[0]):
                inst_id = int(qry_inst['inst_id'][i].numpy())
                orig_img, inst_img, inst_mask = test_dataset.getInstByIDForTest(inst_id)
                inst_mask[inst_mask == 1] = 255
                tmp_pred_mask = PA_qry_mask[i]
                tmp_pred_mask[tmp_pred_mask == 1] = 255
                tmp_pred_mask = cv2.resize(PA_qry_mask[i], (inst_img.shape[1], inst_img.shape[0]))
                cv2.imshow('orig_img', orig_img)
                cv2.imshow('inst_img', inst_img)
                cv2.imshow('gt_mask', inst_mask)
                cv2.imshow('pred_mask', tmp_pred_mask)
                cv2.waitKey()

            intersects = np.sum(np.logical_and(PA_qry_mask == 1, gt_mask == 1), axis=(1,2))
            unions = np.sum(np.logical_or(PA_qry_mask == 1, gt_mask == 1), axis=(1,2))
            weights = np.array(qry_inst['inst_shape'][:,0]*qry_inst['inst_shape'][:,1]) / (opts['roi_size']*opts['roi_size'])
            intersect_sum += np.sum(intersects * weights)
            union_sum += np.sum(unions * weights)

    print(intersect_sum / union_sum)


if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    opts = json.load(open('./config.json', 'r'))
    torch.cuda.set_device(device=opts['main_gpu'])
    test(opts)
