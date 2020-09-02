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


def train(opts):
    # Prepare data
    val_dataset = FewShotInstData(opts, subset='val2017', folds=[3])
    val_loader = DataLoader(val_dataset, batch_size=opts['batch_size'], num_workers=opts['num_workers'], shuffle=False, pin_memory=True)
    train_dataset = FewShotInstData(opts, subset='train2017', folds=[0,1,2])
    train_loader = DataLoader(train_dataset, batch_size=opts['batch_size'], num_workers=opts['num_workers'], shuffle=True, pin_memory=True)

    model = FewShotNet(opts).cuda()
    model = nn.DataParallel(model, device_ids=opts['gpu_list'])
    optimizer = torch.optim.SGD(model.parameters(), lr=opts['lr'], momentum=opts['momentum'], weight_decay=opts['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 30000], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Begin training
    train_loss = 0.0
    best_iou = 0.0
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        qry_inst, supp_insts = batch_data

        PA_qry_pred, PA_align_loss = model(qry_inst, supp_insts)
        PA_qry_loss = criterion(PA_qry_pred, qry_inst['roi_mask'].long().cuda())

        loss = PA_qry_loss + PA_align_loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.detach().data.cpu().numpy()
        if (i + 1) % opts['print_freq'] == 0:
            train_loss /= (opts['print_freq']*opts['batch_size'])
            print(f'Step: {i+1}, Average loss: {train_loss}')
            train_loss = 0.0
        if (i + 1) % opts['val_freq'] == 0:
            print('Validating...')
            model.eval()
            val_iou = val(opts, model, val_loader)
            print('Best IoU: %.3f, Current IoU: %.3f' % (best_iou, val_iou))
            model.train()

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), opts['ckp_dir']+'/best.pth')
                print('Model saved')


def val(opts, model, val_loader):
    """
    Validate on unseen instances in COCO val2017.
    """
    iou = 0.0
    intersect_sum, union_sum = 0.0, 0.0
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            qry_inst, supp_insts = batch_data

            PA_qry_pred, PA_align_loss = model(qry_inst, supp_insts)
            PA_qry_mask = np.array(PA_qry_pred.argmax(dim=1).cpu()).astype(np.uint8)
            gt_mask = np.array(qry_inst['roi_mask']).astype(np.uint8)

            intersects = np.sum(np.logical_and(PA_qry_mask == 1, gt_mask == 1), axis=(1,2))
            unions = np.sum(np.logical_or(PA_qry_mask == 1, gt_mask == 1), axis=(1,2))
            weights = np.array(qry_inst['inst_shape'][:,0]*qry_inst['inst_shape'][:,1]) / (opts['roi_size']*opts['roi_size'])
            intersect_sum += np.sum(intersects * weights)
            union_sum += np.sum(unions * weights)

    return intersect_sum / union_sum


if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    opts = json.load(open('./config.json', 'r'))
    torch.cuda.set_device(device=opts['main_gpu'])
    train(opts)
