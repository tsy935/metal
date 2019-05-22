from torch.utils.data import Dataset, DataLoader
from mapillary_utils import get_file_names, load_labels_and_label_file, get_imagenet_transform
from torchvision import transforms, utils
from PIL import Image
import os
import numpy as np
import torch
class MapillaryDataset(Dataset):
    def __init__(self, root_dir, split='train', input_transform=None, label_transform=None, active_slice_heads={}, overfit_on_slice=None):
        # load the label pickles
        self.root_dir = root_dir
        self.file_names, self.images_dir = get_file_names(self.root_dir, split)
        self.labels_file, self.label_mappings = load_labels_and_label_file(self.root_dir, split)
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.slices = {}
        self.labelsets = {}
        self.active_slice_heads = active_slice_heads
        self.overfit_on_slice = overfit_on_slice
        # must be done *after* defining the file_names
        self.compute_slices()
        self.get_labelsets()
    def compute_slices(self):
        for label in self.label_mappings:
            mask = []
            files_with_label = set(self.label_mappings[label])
            for f in self.file_names:
                if f in files_with_label:
                    mask.append(1)
                else:
                    mask.append(0)
            self.slices[label] = torch.tensor(mask)
    def get_labelsets(self):
        for sname, mask in self.slices.items():
            if self.active_slice_heads.get("pred"):
                slice_labelset_name = f"labelset:{sname}:pred"
                self.labelsets[slice_labelset_name] = mask
            if self.active_slice_heads.get("ind"):
                slice_labelset_name = f"labelset:{sname}:pred"
                mask[mask == 0] = 2 #metal convention
                self.labelsets[slice_labelset_name] = mask
            if self.active_slice_heads.get("shared_pred"):
                slice_labelset_name = f"labelset:{sname}:shared_pred"
                self.labelsets[slice_labelset_name] = mask
            if self.overfit_on_slice is not None:
                slice_labelset_name = f"labelset:{sname}:pred"
                self.labelsets[slice_labelset_name] = mask

    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.images_dir, self.file_names[idx]))
        if self.input_transform:
            im = self.input_transform(im)
        label = self.labels_file[self.file_names[idx]]
        if self.label_transform:
            label = self.label_transform(label)
        
        x_dict = {'data' : im}
        y_dict = {ls_name : mask[idx] if 'ind' in ls_name else mask[idx]*label for ls_name, mask in self.labelsets.items()}
        y_dict = {'labelset_gold': torch.tensor(label)}
    
        return x_dict, y_dict

# split \in [train, val, test]
def get_mapillary_dataset(root_dir, binary_category = 'human--person', split='train'):
    train_transform, val_transform = get_imagenet_transform()
    input_transform = train_transform if split=='train' else val_transform
    def label_transform(labels):
        for label in labels:
            if label[1] == binary_category:
                return 1
        return 2
    return MapillaryDataset(root_dir, split, input_transform=input_transform, label_transform=label_transform)
