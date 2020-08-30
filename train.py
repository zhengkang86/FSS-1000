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
from network.fs_ctr_net import FSCtrNet
from utils import EvalIoU


def train(opts):
    # Prepare data
    train_dataset = FewShotInstData(coco_dir='/WD1/few-shot/ms-coco', subset='train2017', folds=[0,1,2], n_shots=opts['n_shots'])
    train_loader = DataLoader(train_dataset, batch_size=opts['batch_size'], num_workers=opts['num_workers'], shuffle=True)
    val_dataset = FewShotInstData(coco_dir='/WD1/few-shot/ms-coco', subset='val2017', folds=[3], n_shots=opts['n_shots'])
    val_loader = DataLoader(val_dataset, batch_size=opts['batch_size'], num_workers=opts['num_workers'], shuffle=False)

    model = FSCtrNet(opts)
    model = nn.DataParallel(model.cuda(), device_ids=[opts['main_gpu']])
    optimizer = torch.optim.SGD(model.parameters(), lr=opts['lr'], momentum=opts['momentum'], weight_decay=opts['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 30000], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Begin training
    train_loss = 0.0
    for i, batch_data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        query_inst, supp_insts = batch_data
        PA_query_pred, PA_align_loss = model(query_inst, supp_insts)
        PA_query_loss = criterion(PA_query_pred, query_inst['roi_mask'].long().cuda())
        loss = PA_query_loss + PA_align_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # inst_id = query_inst['inst_id'].detach().data.cpu().numpy()
        train_loss += loss.detach().data.cpu().numpy()
        if (i + 1) % opts['print_freq'] == 0:
            train_loss /= (opts['print_freq']*opts['batch_size'])
            print(f'Step: {i+1}, Average loss: {train_loss}')
            train_loss = 0.0
        if (i + 1) % opts['val_freq'] == 0:
            print('Validating...')
            val_iou = val(opts, model, val_loader)
            print(f'Validation IoU: {val_iou}')


def val(opts, model, val_loader):
    """
    Validate on 1000 unseen instances in COCO val2017.
    """
    model.eval()
    iou = 0.0
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            query_inst, supp_insts = batch_data
            PA_query_pred, PA_align_loss = model(query_inst, supp_insts)
            PA_query_mask = np.array(PA_query_pred.argmax(dim=1).cpu()).astype(np.uint8)
            gt_mask = np.array(query_inst['roi_mask']).astype(np.uint8)
            for k in range(gt_mask.shape[0]):
                iou += EvalIoU(PA_query_mask[k], gt_mask[k])
    return iou/val_loader.dataset.__len__()


if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    opts = json.load(open('./config.json', 'r'))
    torch.cuda.set_device(device=opts['main_gpu'])
    train(opts)
