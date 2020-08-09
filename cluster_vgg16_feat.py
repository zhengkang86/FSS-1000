import os
import glob
import tqdm

import sklearn
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, vgg16_bn


class VOCDataset(Dataset):
    def __init__(self, image_list, transforms):
        self.image_list = image_list
        self.transforms = transforms

    def __getitem__(self, idx):
        fname = os.path.splitext(os.path.basename(self.image_list[idx]))[0]
        image = Image.open(self.image_list[idx])
        data_dict = {'fname': fname, 'image': image}
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __len__(self):
        return len(self.image_list)


class TransformCollections(object):
    def __init__(self, field):
        self.field = field
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize((224, 224))
        self.transforms = transforms.Compose([self.resize,
                                              transforms.ToTensor(),
                                              self.normalize,
                                              ])

    def __call__(self, data_dict):
        image = data_dict[self.field]
        image = self.transforms(image)
        data_dict[self.field] = image
        return data_dict


def extract_vgg16_feature(image_list, feat_dir, feature_level=5):
    if not os.path.isdir(feat_dir):
        os.makedirs(feat_dir)

    # create dataloader
    val_transforms = TransformCollections(field='image')
    voc_dataset = VOCDataset(image_list, val_transforms)
    val_loader = DataLoader(voc_dataset, batch_size=16, num_workers=12, shuffle=False, pin_memory=True)

    # create model
    model = vgg16_bn(pretrained=True)
    model = model.cuda()
    model = model.eval()

    # inference
    device = torch.cuda.current_device()
    for idx, batch_data in enumerate(tqdm.tqdm(val_loader)):
        fnames = batch_data['fname']
        images = batch_data['image']
        images = images.to(device)
        # feat = model(images)
        feat = model.features(images)
        feat = model.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = model.classifier[0:feature_level](feat)

        for j in range(feat.shape[0]):
            feat_file = os.path.join(feat_dir, fnames[j])
            feat_j = feat[j]

            import pdb; pdb.set_trace()


def crop_by_bbox(image_dir, anno_dir, seg_dir, seg_list=None, image_ext='jpg', anno_ext='xml'):
    """
    Crop VOC images and segmentations by bounding boxes

    Input:
        image_dir: str, directory of VOC images
        anno_dir: str, directory of VOC annotations
        seg_dir: str, directory of VOC image segmentations (instaces)
    Return:
        None
    """
    roi_dir = os.path.dirname(image_dir) + '/ROIs'
    seg_roi_dir = os.path.dirname(image_dir) + '/Seg_ROIs'
    if not os.path.isdir(roi_dir):
        os.makedirs(roi_dir)
    if not os.path.isdir(seg_roi_dir):
        os.makedirs(seg_roi_dir)

    image_list = glob.glob(image_dir + '/*.' + image_ext)
    for image_file in image_list:
        basename = os.path.basename(image_file)
        fname = os.path.splitext(basename)[0]
        if fname not in seg_list:
            continue

        anno_file = os.path.join(anno_dir, fname + '.' + anno_ext)
        root = ET.parse(anno_file).getroot()

        boxes = {}
        for obj in root.iter('object'):
            filename = root.find('filename').text
            ymin, xmin, ymax, xmax = None, None, None, None
            cls_name = obj.find('name').text.strip().lower()

            all_boxes = obj.findall('bndbox')
            assert(len(all_boxes) == 1)
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            boxes[cls_name] = []
            boxes[cls_name].append([xmin, ymin, xmax, ymax])

        import pdb; pdb.set_trace()


if __name__ == '__main__':
    voc_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012'
    image_dir = os.path.join(voc_dir, 'JPEGImages')

    """
    # crop VOC images and segmentations based on bounding boxe
    """
    # anno_dir = os.path.join(voc_dir, 'Annotations')
    # seg_dir = os.path.join(voc_dir, 'Segmentation_Objects')
    # seg_list_file = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    # with open(seg_list_file, 'r') as f:
    #     seg_list = f.read().splitlines()

    # crop_by_bbox(image_dir, anno_dir, seg_dir, seg_list)

    """
    # extract vgg16 features for VOC images
    """
    image_list = glob.glob(image_dir + '/*.jpg')
    feat_dir = os.path.join(voc_dir, 'features')
    extract_vgg16_feature(image_list, feat_dir, feature_level=3)
