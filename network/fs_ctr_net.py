import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .vgg import Encoder
from .PANet import forward as PANetForward


def MakePANetInput(query_img, supp_imgs, supp_masks, query_ft, flat_supp_fts):
    PA_qry_imgs = [query_img]
    PA_supp_imgs = [[supp_imgs[i] for i in range(supp_imgs.shape[0])]]
    bg_mask = 1 - supp_masks
    PA_fore_mask = [[supp_masks[i] for i in range(supp_masks.shape[0])]]
    PA_back_mask = [[bg_mask[i] for i in range(bg_mask.shape[0])]]
    PA_img_fts = torch.cat([flat_supp_fts, query_ft], dim=0)
    return PA_supp_imgs, PA_fore_mask, PA_back_mask, PA_qry_imgs, PA_img_fts


class FSCtrNet(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels=3, pretrained_path=opts['pretrained_vgg']))]))

    def forward(self, query_inst, supp_insts):
        batch_size = query_inst['roi_img'].shape[0]
        n_shot = len(supp_insts)

        query_img = query_inst['roi_img']
        query_mask = query_inst['roi_mask']
        query_init_ctr = query_inst['init_ctr']
        query_gt_ctr = query_inst['gt_ctr']

        def reshapeSupp(key):
            supp_value = torch.stack([supp_insts[i][key] for i in range(n_shot)])
            return supp_value

        supp_imgs = reshapeSupp('roi_img')  # batch_size*n_shot*C*H*W
        supp_masks = reshapeSupp('roi_mask')
        supp_init_ctrs = reshapeSupp('init_ctr')
        supp_gt_ctrs = reshapeSupp('gt_ctr')

        query_ft = self.encoder(query_img)
        flat_supp_imgs = supp_imgs.reshape(1, -1, *(supp_imgs.size()[2:]))[0]  # Flatten supp_imgs for encoding (supp_imgs[0][1]==flat_supp_imgs[1])
        flat_supp_fts = self.encoder(flat_supp_imgs)

        PA_supp_imgs, PA_fore_mask, PA_back_mask, PA_qry_imgs, PA_img_fts = MakePANetInput(query_img, supp_imgs, supp_masks, query_ft, flat_supp_fts)
        PA_query_pred, PA_align_loss = PANetForward(PA_supp_imgs, PA_fore_mask, PA_back_mask, PA_qry_imgs, PA_img_fts)

        return PA_query_pred, PA_align_loss
