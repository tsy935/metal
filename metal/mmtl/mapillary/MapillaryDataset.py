from torch.utils.data import Dataset, DataLoader
from mapillary_utils import get_file_names, load_labels_and_label_file, get_imagenet_transform
from torchvision import transforms, utils
from PIL import Image
class MapillaryDataset(Dataset):
    def __init__(self, root_dir, split='train', input_transform=None, label_transform=None):
        # load the label pickles
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'training', 'images')
        self.labels_file, self.label_mappings = load_labels_and_label_file(split)
        self.file_names = get_file_names(self.root_dir, split)
        self.input_transform = input_transform
        self.label_transform = label_transform
    def __len__(self):
        return len(self.labels_file)
    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.images_dir, self.file_names[idx]))
        if self.input_transform:
            im = self.input_transform(im)
        labels = self.labels_file[self.file_names[idx]]
        if self.label_transform:
            labels = self.label_transform(labels)
        return im, labels
    
def get_mapillary_dataset(root_dir, binary_category = 'human--person', split='train'):
    train_transform, val_transform = get_imagenet_transform()
    input_transform = train_transform if split=='train' else val_transform
    def label_transform(labels):
        for label in labels:
            if label[1] == binary_category:
                return 1
        return 0
    return MapillaryDataset(root_dir, split, input_transform=input_transform, label_transform=label_transform)
