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

CLASS_LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CLASS_COLOR_MAP = np.asarray([[0, 0, 0],
                              [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                              [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                              [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                              [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]])


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
    roi_dir = os.path.dirname(image_dir) + '/val_ROIs'
    seg_roi_dir = os.path.dirname(image_dir) + '/val_Seg_ROIs'
    if not os.path.isdir(roi_dir):
        os.makedirs(roi_dir)
    if not os.path.isdir(seg_roi_dir):
        os.makedirs(seg_roi_dir)

    class_id = {class_name: i + 1 for i, class_name in enumerate(CLASS_LIST)}

    image_list = glob.glob(image_dir + '/*.' + image_ext)
    for image_file in tqdm.tqdm(image_list):
        basename = os.path.basename(image_file)
        fname = os.path.splitext(basename)[0]
        if fname not in seg_list:
            continue
        seg_file = os.path.join(seg_dir, fname + '.png')
        image = Image.open(image_file).convert('RGB')
        image = np.array(image)
        seg_rgb = Image.open(seg_file).convert('RGB')
        seg_rgb = np.array(seg_rgb)

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

            cid = class_id[cls_name]
            r, g, b = CLASS_COLOR_MAP[cid]
            seg = seg_rgb.astype(np.float32)
            channel_r = (seg[:, :, 0] == r).astype(np.uint8)
            channel_g = (seg[:, :, 1] == g).astype(np.uint8)
            channel_b = (seg[:, :, 2] == b).astype(np.uint8)
            seg = (channel_r * channel_g) * channel_b

            if seg.max() != 1:
                print(seg.max(), image_file, seg_file)
                continue
            assert(seg.max() == 1)

            # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # ax[0].imshow(image)
            # ax[0].set_title('class {}: {}'.format(cid, cls_name))
            # ax[1].imshow(seg_rgb)
            # ax[1].set_title('class cmap: [{}, {}, {}]'.format(r, g, b))
            # ax[2].imshow(seg)
            # plt.show()

            seg_roi = seg[ymin:ymax, xmin:xmax]
            seg_roi_file = os.path.join(seg_roi_dir, '{}#{}_{}.png'.format(fname, cls_name, cls_cnt[cls_name]))
            seg_roi = Image.fromarray(seg_roi)
            seg_roi.save(seg_roi_file)


def cluster_vgg16_feat():
    feat_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/ROIs_vgg16_feat'
    roi_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/ROIs'
    centers_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/kmeans_centers'
    if not os.path.isdir(centers_dir):
        os.makedirs(centers_dir)

    for cls_name in CLASS_LIST:
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
        center_image_list = [os.path.join(roi_dir, os.path.basename(center_feat_file).replace('.npy', '.jpg')) for
                             center_feat_file in center_feat_list]

        dst_dir = os.path.join(centers_dir, cls_name)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        for center_image_file in center_image_list:
            shutil.copy(center_image_file, dst_dir)


if __name__ == '__main__':
    voc_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012'

    """
    # crop VOC images and segmentations based on bounding boxe
    """
    # image_dir = os.path.join(voc_dir, 'JPEGImages')
    # anno_dir = os.path.join(voc_dir, 'Annotations')
    # seg_dir = os.path.join(voc_dir, 'SegmentationClass')
    # seg_list_file = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    # # seg_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/Binary_map_aug/train/'
    # with open(seg_list_file, 'r') as f:
    #     seg_list = f.read().splitlines()
    # crop_by_bbox(image_dir, anno_dir, seg_dir, seg_list)

    """
    # extract vgg16 features for VOC images
    """
    # roi_dir = os.path.join(voc_dir, 'ROIs')
    # image_list = glob.glob(roi_dir + '/*.jpg')
    # image_list = sorted(image_list)
    # feat_dir = os.path.join(voc_dir, 'ROIs_vgg16_feat')
    # extract_vgg16_feature(image_list, feat_dir, classifier_layer=5)

    """
    cluster vgg16 features using k-means
    """
    # cluster_vgg16_feat()

    """
    copy GT segmentation masks to support set
    """
    # support_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/fss/support/'
    # support_image_dir = os.path.join(support_dir, 'images')
    # support_label_dir = os.path.join(support_dir, 'labels')
    # seg_roi_dir = '/home/kang/Projects/data/VOC/VOCdevkit/VOC2012/Seg_ROIs'
    # for cls in CLASS_LIST:
    #     cls_image_dir = os.path.join(support_image_dir, cls)
    #     image_files = glob.glob(cls_image_dir + '/*.jpg')
    #     cls_label_dir = os.path.join(support_label_dir, cls)
    #     if not os.path.isdir(cls_label_dir):
    #         os.mkdir(cls_label_dir)
    #     for image_file in image_files:
    #         fname = os.path.splitext(os.path.basename(image_file))[0]
    #         src_label_file = os.path.join(seg_roi_dir, fname + '.png')
    #         shutil.copy(src_label_file, cls_label_dir)

    """
    move images
    """
    query_dir = os.path.join(voc_dir, 'fss', 'query', 'images')
    for cls in CLASS_LIST:
        cls_dir = os.path.join(query_dir, cls)
        if not os.path.isdir(cls_dir):
            os.mkdir(cls_dir)
        image_files = glob.glob(query_dir + f'/*{cls}*.jpg')
        for image_file in image_files:
            shutil.move(image_file, cls_dir)
