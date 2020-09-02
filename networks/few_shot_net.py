import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .vgg import Encoder
from .PANet import forward as PANetForward


def MakePANetInput(qry_img, supp_imgs, supp_masks, qry_ft, flat_supp_fts):
    PA_qry_imgs = [qry_img]
    PA_supp_imgs = [[supp_imgs[i] for i in range(supp_imgs.shape[0])]]
    bg_mask = 1 - supp_masks
    PA_fore_mask = [[supp_masks[i] for i in range(supp_masks.shape[0])]]
    PA_back_mask = [[bg_mask[i] for i in range(bg_mask.shape[0])]]
    PA_img_fts = torch.cat([flat_supp_fts, qry_ft], dim=0)
    return PA_supp_imgs, PA_fore_mask, PA_back_mask, PA_qry_imgs, PA_img_fts


class FewShotNet(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels=3, pretrained_path=opts['pretrained_vgg']))]))

    def forward(self, qry_inst, supp_insts):
        batch_size = qry_inst['roi_img'].shape[0]
        n_shot = len(supp_insts)

        qry_img = qry_inst['roi_img'].cuda()
        qry_mask = qry_inst['roi_mask'].cuda()
        qry_init_ctr = qry_inst['init_ctr'].cuda()
        qry_gt_ctr = qry_inst['gt_ctr'].cuda()

        def reshapeSupp(key):
            supp_value = torch.stack([supp_insts[i][key] for i in range(n_shot)])
            return supp_value.cuda()

        supp_imgs = reshapeSupp('roi_img')  # batch_size*n_shot*C*H*W
        supp_masks = reshapeSupp('roi_mask')
        supp_init_ctrs = reshapeSupp('init_ctr')
        supp_gt_ctrs = reshapeSupp('gt_ctr')

        qry_ft = self.encoder(qry_img)
        flat_supp_imgs = supp_imgs.reshape(1, -1, *(supp_imgs.size()[2:]))[0]  # Flatten supp_imgs for encoding (supp_imgs[0][1]==flat_supp_imgs[1])
        flat_supp_fts = self.encoder(flat_supp_imgs)

        PA_supp_imgs, PA_fore_mask, PA_back_mask, PA_qry_imgs, PA_img_fts = MakePANetInput(qry_img, supp_imgs, supp_masks, qry_ft, flat_supp_fts)
        PA_qry_pred, PA_align_loss = PANetForward(PA_supp_imgs, PA_fore_mask, PA_back_mask, PA_qry_imgs, PA_img_fts)

        # output = self.curve_gcn(qry_img, qry_init_ctr)
        # pred_poly = output['pred_polys'][-1]

        return PA_qry_pred, PA_align_loss
