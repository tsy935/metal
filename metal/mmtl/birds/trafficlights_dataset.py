import copy
import torch.utils.data as data
import json
import os
import torch
# sys.path.append('/dfs/scratch0/zzweng/metal/metal/mmtl/birds')
os.environ['METALHOME'] = '/dfs/scratch0/zzweng/metal'
from skimage import io, transform
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from metal.utils import set_seed
from metal.mmtl.payload import Payload
from metal.mmtl.data import MmtlDataLoader, MmtlDataset
from metal.mmtl.slicing.tasks import MultiClassificationTask
from metal.mmtl.birds.resnet import resnet18
from metal.mmtl.slicing.tasks import create_slice_task

DATASET_DIR = '/dfs/scratch0/chami/maskrcnn-benchmark/datasets/traffic_light_data/'
IMAGES_DIR = os.path.join(DATASET_DIR, 'frames')
save_dir = '/dfs/scratch0/chami/maskrcnn-benchmark/datasets/traffic_light_data/annotations/'
tt = transforms.ToTensor()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
task_defaults = {
	"active_slice_heads": {"ind": True, "pred": True},
	"seed" : None,
	'batch_size' : 32,
	'overfit_on_slice' : None,
}

class Dataset(data.Dataset):
    def __init__(self, split, slice_names, active_slice_heads):
        with open(os.path.join(save_dir, f'{split}.json')) as f:
            annotations = json.load(f)['annotations']
        self.filenames = {}
        self.labels = {}
        self.bbox = {}
        self.slice_names = slice_names
        self.active_slice_heads = active_slice_heads
        self.is_night = {}
        self.is_day = {}
        self.is_yellow = {}
        for a in annotations:
            img_id = a['image_id']
            self.filenames[img_id] = a['filename']
            if img_id not in self.labels or self.labels[img_id] != 1:
                self.labels[img_id] = 2 - int(
                    a['tag'] in ['stop', 'warning'])  # label = 1 if red(stop) or yellow(warning)
            if img_id not in self.bbox:
                self.bbox[img_id] = [a['bbox']]
            else:
                self.bbox[img_id].append(a['bbox'])
            self.is_night[img_id] = int("night" in a['filename'])  # night -> 1
            self.is_day[img_id] = int("day" in a['filename'])  # day -> 1
            self.is_yellow[img_id] = int(a['tag'] == 'warning')  # yellow light -> 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx += 1  # file IDs starts with 1
        fname = self.filenames[idx]
        image_data = io.imread(os.path.join(IMAGES_DIR, fname))
        image_data = transform.resize(image_data, (224, 224, 3))  # resize all images to 224x224
        image_data = normalize(tt(image_data).type(torch.float32))

        x_dict = {"data": image_data}
        is_night_mask = self.is_night[idx]
        is_day_mask = self.is_day[idx]
        is_yellow_mask = self.is_yellow[idx]

        # generating labels
        y_dict = {"labelset_gold": torch.tensor([self.labels[idx]])}
        if self.active_slice_heads.get("pred"):
            y_dict.update({
                "labelset:is_night_slice:pred": torch.tensor([self.labels[idx] if is_night_mask else 0]),
                "labelset:is_day_slice:pred": torch.tensor([self.labels[idx] if is_day_mask else 0]),
                "labelset:is_yellow_slice:pred": torch.tensor([self.labels[idx] if is_yellow_mask else 0]),
            })
        if self.active_slice_heads.get("ind"):
            y_dict.update({
                "labelset:is_night_slice:ind": torch.tensor([1 if is_night_mask else 2]),
                "labelset:is_day_slice:ind": torch.tensor([1 if is_day_mask else 2]),
                "labelset:is_yellow_slice:ind": torch.tensor([1 if is_yellow_mask else 2])
            })
        if self.active_slice_heads.get("shared_pred"):
            y_dict.update({
                "labelset:is_night_slice:shared_pred": torch.tensor([self.labels[idx] if is_night_mask else 0]),
                "labelset:is_day_slice:shared_pred": torch.tensor([self.labels[idx] if is_day_mask else 0]),
                "labelset:is_yellow_slice:shared_pred": torch.tensor([self.labels[idx] if is_yellow_mask else 0]),
            })
        # TODO: add overfit_on_slice stuff.

        return x_dict, y_dict

    def draw(self, image_id):
        fname = self.filenames[image_id]
        print('Visualizing: ', fname)
        img = Image.open(os.path.join(IMAGES_DIR, fname))
        draw = ImageDraw.Draw(img)
        for (left, bottom, right, top) in self.bbox[image_id]:
            draw.rectangle([left, bottom, right, top], outline='red', width=3)
        return img


def create_traffic_lights_tasks_payloads(slice_names, **task_config):
    set_seed(task_config['seed'])
    active_slice_heads = task_config['active_slice_heads']

    resnet_model = resnet18(num_classes=2, use_as_feature_extractor=True).float().cuda()
    task_name = 'TrafficLightClassificationTask'
    task0 = MultiClassificationTask(
        name=task_name,
        input_module=resnet_model,
        head_module=resnet_model.fc
    )
    tasks = [task0]
    slice_names = slice_names + ['BASE']

    if task_config['overfit_on_slice'] != None:
        slice_names.append(int(task_config['overfit_on_slice']))

    # TODO: why?
    loss_multiplier = 1.0 / (2 * (len(slice_names)))  # +1 for Base

    if task_config['active_slice_heads'].get('shared_pred'):
        shared_pred_head_module = copy.deepcopy(task0.head_module)

    # Generate slice tasks
    for attr_id in slice_names:
        if task_config['active_slice_heads'].get('pred'):
            slice_task_name = f"{task_name}:{attr_id}:pred"
            slice_task = create_slice_task(task0,
                                           slice_task_name,
                                           slice_head_type='pred',
                                           loss_multiplier=loss_multiplier,
                                           )
            tasks.append(slice_task)

        # for ind slice head type
        if task_config['active_slice_heads'].get('ind'):
            slice_task_name = f"{task_name}:{attr_id}:ind"
            slice_task = create_slice_task(task0,
                                           slice_task_name,
                                           slice_head_type='ind',
                                           loss_multiplier=loss_multiplier,
                                           )
            tasks.append(slice_task)

        if task_config['active_slice_heads'].get('shared_pred'):
            slice_task = copy.copy(task0)
            slice_task.name = f"{task_name}:{attr_id}:shared_pred"
            slice_task.slice_head_type = 'pred'
            slice_task.head_module = shared_pred_head_module
            tasks.append(slice_task)

    payloads = []
    splits = ["train", "valid", "test"]
    annotation_file_name = ["train", "val", "test"]
    splits_shuffle = [True, False, False]
    datasets = []
    for i, split in enumerate(splits):
        payload_name = f"Payload{i}_{split}"
        ds = Dataset(annotation_file_name[i], slice_names, active_slice_heads)
        datasets.append(ds)
        labels_to_tasks = {"labelset_gold": task_name}
        for attr_id in slice_names:
            if task_config['overfit_on_slice'] != None and attr_id == int(task_config['overfit_on_slice']):
                s = task_config['overfit_on_slice']
                slice_labelset_name = f"labelset:{s}:pred"
                slice_task_name = f"{task_name}:{s}:pred"
                labels_to_tasks[slice_labelset_name] = slice_task_name

            if active_slice_heads.get("pred"):
                slice_labelset_name = f"labelset:{attr_id}:pred"
                slice_task_name = f"{task_name}:{attr_id}:pred"
                labels_to_tasks[slice_labelset_name] = slice_task_name

            if active_slice_heads.get('shared_pred'):
                slice_labelset_name = f"labelset:{attr_id}:shared_pred"
                slice_task_name = f"{task_name}:{attr_id}:shared_pred"
                labels_to_tasks[slice_labelset_name] = slice_task_name

            # for shared pred head type
            if active_slice_heads.get('shared_pred'):
                slice_labelset_name = f"labelset:{attr_id}:shared_pred"
                slice_task_name = f"{task_name}:{attr_id}:shared_pred"
                labels_to_tasks[slice_labelset_name] = slice_task_name

            # for ind slice head type
            if task_config['active_slice_heads'].get('ind'):
                slice_labelset_name = f"labelset:{attr_id}:ind"
                slice_task_name = f"{task_name}:{attr_id}:ind"
                labels_to_tasks[slice_labelset_name] = slice_task_name

        payload = Payload(
            payload_name,
            MmtlDataLoader(datasets[i], shuffle=splits_shuffle[i], batch_size=task_config['batch_size']),
            labels_to_tasks,
            split
        )

        if task_config['overfit_on_slice'] != None:
            payload.remap_labelsets({'labelset:{}:pred'.format(task_config['overfit_on_slice']): task_name},
                                    default_none=True)

        payloads.append(payload)

    return tasks, payloads
