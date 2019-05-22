import os
import json
import pickle
import numpy as np
from torchvision import transforms, utils
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from PIL import Image

def get_imagenet_transform(normalize=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = []
    train_transform.extend([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    if normalize:
        train_transform.append(normalize)
    train_transform = transforms.Compose(train_transform)
    val_transform = []
    val_transform.extend([
        transforms.Resize(229),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    if normalize:
        val_transform.append(normalize)
    val_transform = transforms.Compose(val_transform)
    return train_transform, val_transform

def get_file_names(root_dir, split='train'):
    train_dir = os.path.join(root_dir, 'training', 'images')
    val_dir = os.path.join(root_dir, 'validation', 'images')
    if split == 'train':
        ims = os.listdir(train_dir)
    else:
        ims = os.listdir(val_dir)
    return ims

def get_all_labels(root_dir):
    with open(os.path.join(ROOT_DIR, 'config.json')) as config_file:
        config = json.load(config_file)
    all_labels = config['labels']
    return all_labels

def get_labels(image_array, all_labels):
    labels = []
    for label_id, label in enumerate(all_labels):
        presence_mask = image_array == label_id
        if np.sum(presence_mask) != 0:
            labels.append((label['readable'], label['name'], label['instances']))
    return labels

def get_label_to_file(split_labels):
    label_to_file = {}
    label_len = 0
    for f_name, labels in split_labels.items():
        label_len += len(labels)
        for l in labels:
            if l[1] not in label_to_file:
                label_to_file[l[1]] = []
            label_to_file[l[1]].append(f_name)
    print('Average label size: {}'.format(label_len/len(split_labels)))
    return label_to_file

def save_dict(obj, f_name):
    with open(f_name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_labels_and_label_file(split='train'):
    if split == 'val':
        f_labels_name, f_label_to_file_name = 'val_labels.pickle', 'val_label_to_file.pickle'
    else:
        f_labels_name, f_label_to_file_name = 'train_labels.pickle', 'train_label_to_file.pickle'
    with open(f_labels_name, 'rb') as in_f:
        labels = pickle.load(in_f)
    with open(f_label_to_file_name, 'rb') as in_f:
        label_to_file = pickle.load(in_f)
    return labels, label_to_file