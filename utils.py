import os
import cv2
import numpy as np


def CheckContourIfSorted(img, ctr):
    '''
    Connect each point to the next, to see if they are sorted.
    :param img: Use its shape only
    :param ctr: a N*2 array
    :return:
    '''
    tmp_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(0, len(ctr)-2):
        cv2.line(tmp_img, tuple(ctr[i]), tuple(ctr[i+1]), 255, 1)
    cv2.line(tmp_img, tuple(ctr[-1]), tuple(ctr[0]), 255, 1)
    cv2.imshow('tmp_img', tmp_img)
    cv2.waitKey()


def ShowCtrPointByPoint(img, ctr):
    '''
    Draw a contour point by point.
    :param img: Use its shape only
    :param ctr: a N*2 array
    :return:
    '''
    tmp_img = np.zeros(img.shape, dtype=np.uint8)
    for p in ctr:
        tmp_img[p[1], p[0]] = 255
        cv2.imshow('tmp_img', tmp_img)
        cv2.waitKey()


def UniformSampleCtr(pgtnp_px2, newpnum):
    '''
    Sample a sorted contour to a certain length. (Copied from Curve-GCN)
    :param pgtnp_px2: sorted contour
    :param newpnum: new length
    :return: sampled contour
    '''
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def EvalIoU(pred, gt, num_classes=2):
    """
    Deprecated.
    """
    iou = 0.0
    for i in range(1, num_classes):
        intersection = np.sum(np.logical_and(pred == i, gt == i))
        union = np.sum(np.logical_or(pred == i, gt == i))
        iou += float(intersection)/union

    # Visualization
    # pred[pred == 1] = 255
    # gt[gt == 1] = 255
    # cv2.imshow('pred', pred)
    # cv2.imshow('gt', gt)
    # cv2.waitKey()

    return iou/(num_classes-1)



if __name__ == '__main__':
    pass
    # FindBadInstance()
