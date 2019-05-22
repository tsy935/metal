from torch.utils.data import Dataset, DataLoader
from mapillary_utils import get_file_names, load_labels_and_label_file, get_imagenet_transform
from torchvision import transforms, utils
from PIL import Image
import os
import numpy as np
class MapillaryDataset(Dataset):
    def __init__(self, root_dir, split='train', input_transform=None, label_transform=None):
        # load the label pickles
        self.root_dir = root_dir
        self.file_names, self.images_dir = get_file_names(self.root_dir, split)
        self.labels_file, self.label_mappings = load_labels_and_label_file(self.root_dir, split)
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.slices = {}
        # must be done *after* defining the file_names
        self.compute_slices()
    def compute_slices(self):
        for label in self.label_mappings:
            mask = []
            files_with_label = set(self.label_mappings[label])
            for f in self.file_names:
                if f in files_with_label:
                    mask.append(1)
                else:
                    mask.append(0)
            self.slices[label] = np.array(mask)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.images_dir, self.file_names[idx]))
        if self.input_transform:
            im = self.input_transform(im)
        labels = self.labels_file[self.file_names[idx]]
        if self.label_transform:
            labels = self.label_transform(labels)
        return im, labels

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
