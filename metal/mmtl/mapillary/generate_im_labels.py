import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from PIL import Image
from mapillary_utils import get_labels, get_label_to_file, save_dict
ROOT_DIR = 'mapillary/'
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, 'training', 'images')
TRAIN_LABELS_DIR = os.path.join(ROOT_DIR, 'training', 'labels')
VAL_IMAGES_DIR = os.path.join(ROOT_DIR, 'validation', 'images')
VAL_LABELS_DIR = os.path.join(ROOT_DIR, 'validation', 'labels')
with open(os.path.join(ROOT_DIR, 'config.json')) as config_file:
    config = json.load(config_file)
all_labels = config['labels']

train_ims = os.listdir(TRAIN_IMAGES_DIR)
train_label_ims = os.listdir(TRAIN_LABELS_DIR)
val_ims = os.listdir(VAL_IMAGES_DIR)
val_label_ims = os.listdir(VAL_LABELS_DIR)


def process_train_im_label(i):
    f = train_label_ims[i]
    im = Image.open(os.path.join(TRAIN_LABELS_DIR, f))
    im = np.array(im)
    labels = get_labels(im, all_labels)
    image_path_name = f.split('.')[0] + '.jpg'
    return (image_path_name, labels)

def process_val_im_label(i):
    f = val_label_ims[i]
    im = Image.open(os.path.join(VAL_LABELS_DIR, f))
    im = np.array(im)
    labels = get_labels(im, all_labels)
    image_path_name = f.split('.')[0] + '.jpg'
    return (image_path_name, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-val', action='store_true', help='Generate the val labels instead')
    args = parser.parse_args()
    train_labels = {}
    val_labels = {}
    print('Creating pool')
    pool = Pool(processes=16)
    print('Launching map')
    total_ims = len(val_label_ims) if args.use_val else len(train_label_ims)
    if args.use_val:
        returned_objects = list(tqdm(pool.imap(process_val_im_label, range(total_ims)), total=total_ims))
        f_labels_name, f_label_to_file_name = 'val_labels.pickle', 'val_label_to_file.pickle'
    else:
        returned_objects = list(tqdm(pool.imap(process_train_im_label, range(total_ims)), total=total_ims))
        f_labels_name, f_label_to_file_name = 'train_labels.pickle', 'train_label_to_file.pickle'
    processed_labels = val_labels if args.use_val else train_labels
    for tup in returned_objects:
        f, im_labels = tup
        processed_labels[f] = im_labels
    print('Done with processing labels.')
    print('There are {} files with labels now'.format(len(processed_labels)))
    label_to_file = get_label_to_file(processed_labels)
    save_dict(processed_labels, f_labels_name)
    save_dict(label_to_file, f_label_to_file_name)
