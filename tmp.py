import os
from pycocotools.coco import COCO
import cv2
import skimage.io as io


def PrepCOCO():
    # initialize COCO api for instance annotations
    base_dir = '/WD1/few-shot/ms-coco'
    subset = 'val2017'
    ann_file = '{}/annotations/instances_{}.json'.format(base_dir, subset)
    coco = COCO(ann_file)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    k = 0
    for cat in cats:
        print(cat['name'], cat['id'])
        all_ann_ids = coco.getAnnIds(catIds=cat['id'], iscrowd=False)
        print(len(all_ann_ids))
        k += len(all_ann_ids)
        tmp_ann_id = all_ann_ids[0]
        for tmp_ann_id in all_ann_ids:
            tmp_ann = coco.loadAnns(tmp_ann_id)
            tmp_mask = coco.annToMask(tmp_ann[0])
            tmp_mask[tmp_mask == 1] = 255
            tmp_img_id = tmp_ann[0]['image_id']
            tmp_img = coco.loadImgs(tmp_img_id)[0]
            tmp_img = io.imread(tmp_img['coco_url'])
            print(tmp_ann[0]['iscrowd'])
            cv2.imshow('mask', tmp_mask)
            cv2.imshow('img', tmp_img)
            cv2.waitKey()
    print(k)



if __name__ == '__main__':
    PrepCOCO()
