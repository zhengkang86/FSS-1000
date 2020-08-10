import os
import glob
import tqdm
import shutil

import sklearn
import numpy as np
import matplotlib.pyplot as plt
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


def extract_vgg16_feature(image_list, feat_dir, classifier_layer=5):
    """
    Extracts vgg16 features for images
    Input:
        image_list: list of image paths for feature extraction
        feat_dir: directory to save features
        classifier_layer: which layer the feature is used in classifier
    Output:
        features saved in `feat_dir` in `.npy` format
    """
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

    print(model.classifier[0:classifier_layer])

    # inference
    device = torch.cuda.current_device()
    for idx, batch_data in enumerate(tqdm.tqdm(val_loader)):
        fnames = batch_data['fname']
        images = batch_data['image']
        images = images.to(device)

        feat = model.features(images)
        feat = model.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = model.classifier[0:classifier_layer](feat)

        for j in range(feat.shape[0]):
            feat_file = os.path.join(feat_dir, fnames[j] + '.npy')
            feat_j = feat[j]
            feat_j = feat_j.detach().cpu().numpy()

            with open(feat_file, 'wb') as f:
                np.save(f, feat_j)
            # with open(feat_file, 'rb') as f:
            #     feat_j_reload = np.load(f)


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
    for image_file in tqdm.tqdm(image_list):
        basename = os.path.basename(image_file)
        fname = os.path.splitext(basename)[0]
        if fname not in seg_list:
            continue
        seg_file = os.path.join(seg_dir, fname + '.png')
        image = Image.open(image_file).convert('RGB')
        image = np.array(image)
        seg = Image.open(seg_file).convert('RGB')
        seg = np.array(seg)

        anno_file = os.path.join(anno_dir, fname + '.' + anno_ext)
        root = ET.parse(anno_file).getroot()

        boxes = {}
        cls_cnt = {}
        for obj in root.iter('object'):
            filename = root.find('filename').text
            ymin, xmin, ymax, xmax = None, None, None, None
            cls_name = obj.find('name').text.strip().lower()
            if cls_name in cls_cnt:
                cls_cnt[cls_name] += 1
            else:
                cls_cnt[cls_name] = 1

            all_boxes = obj.findall('bndbox')
            assert(len(all_boxes) == 1)
            xml_box = obj.find('bndbox')
            xmin = int(xml_box.find('xmin').text) - 1
            ymin = int(xml_box.find('ymin').text) - 1
            xmax = int(xml_box.find('xmax').text) - 1
            ymax = int(xml_box.find('ymax').text) - 1
            boxes[cls_name] = []
            boxes[cls_name].append([xmin, ymin, xmax, ymax])

            image_roi = image[ymin:ymax, xmin:xmax, :]
            roi_file = os.path.join(roi_dir, '{}#{}_{}.jpg'.format(fname, cls_name, cls_cnt[cls_name]))
            image_roi = Image.fromarray(image_roi)
            image_roi.save(roi_file)
            
            # seg_roi = seg[ymin:ymax, xmin:xmax]
            # unique_labels, cnt_labels = np.unique(seg_roi, return_counts=True)
            # seg_roi_file = os.path.join(seg_roi_dir, '{}#{}_{}.png'.format(fname, cls_name, cls_cnt[cls_name]))


def cluster_vgg16_feat():
    feat_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/ROIs_vgg16_feat'
    roi_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/ROIs'
    centers_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/kmeans_centers'
    if not os.path.isdir(centers_dir):
        os.path.makedirs(centers_dir)

    class_list = ['person',
                  'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                  'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                  'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    for cls_name in class_list:
        feat_list = glob.glob(feat_dir + f'/*{cls_name}*.npy')
        feat_list = sorted(feat_list)
        feat_dim = 4096
        feat_matrix = np.zeros((len(feat_list), feat_dim))
        for i, feat_file in enumerate(feat_list):
            with open(feat_file, 'rb') as f:
                feat_i = np.load(f)
            feat_matrix[i, :] = feat_i
        
        # k-means clustering
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
        kmeans = KMeans(n_clusters=5, random_state=0).fit(feat_matrix)
        centers = kmeans.cluster_centers_

        nbrs = NearestNeighbors(n_neighbors=1).fit(feat_matrix)
        distances, indices = nbrs.kneighbors(centers)
        
        # copy cluster center images to the centers_dir
        center_feat_list = [feat_list[i] for i in indices.squeeze().tolist()]
        center_image_list = [os.path.join(roi_dir, os.path.basename(center_feat_file).replace('.npy', '.jpg')) for center_feat_file in center_feat_list]
        
        dst_dir = os.path.join(centers_dir, cls_name)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        for center_image_file in center_image_list:
            shutil.copy(center_image_file, dst_dir)



if __name__ == '__main__':
    voc_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012'
    image_dir = os.path.join(voc_dir, 'JPEGImages')

    """
    # crop VOC images and segmentations based on bounding boxe
    """
    anno_dir = os.path.join(voc_dir, 'Annotations')
    seg_dir = os.path.join(voc_dir, 'SegmentationObject')
    seg_list_file = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    with open(seg_list_file, 'r') as f:
        seg_list = f.read().splitlines()

    crop_by_bbox(image_dir, anno_dir, seg_dir, seg_list)

    """
    # extract vgg16 features for VOC images
    """
    roi_dir = os.path.join(voc_dir, 'ROIs')
    image_list = glob.glob(roi_dir + '/*.jpg')
    image_list = sorted(image_list)
    feat_dir = os.path.join(voc_dir, 'ROIs_vgg16_feat')
    extract_vgg16_feature(image_list, feat_dir, classifier_layer=5)

    """
    cluster vgg16 features using k-means
    """
    cluster_vgg16_feat()
