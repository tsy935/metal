
import sys
import metal
import os
import pickle
from metal.mmtl.birds.resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapillary_dataset import MapillaryDataset, get_mapillary_dataset
from metal.mmtl.payload import Payload
from metal.mmtl.data import MmtlDataLoader, MmtlDataset
from pprint import pprint
from metal.mmtl.slicing.tasks import *
from metal.mmtl.metal_model import MetalModel 
from metal.mmtl.trainer import MultitaskTrainer
from metal.utils import set_seed
import pdb
import copy

task_defaults = {
	"active_slice_heads": {"ind": True, "pred": True},
	"seed" : None,
	'batch_size' : 32,
	'overfit_on_slice' : None,
}


'''
[slice_names] is a list of attribute ids
'''
def create_mapillary_tasks_payloads(**task_config):
	set_seed(task_config['seed'])
	
	NUM_CLASSES = 2
	ROOT_DIR = '/home/ankitmathur/mapillary'
	resnet_model = resnet50(use_as_feature_extractor=True, pretrained=True).float().cuda()
	resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
	

	task_name = 'MapillaryClassificationTask'
	task0 = MultiClassificationTask(
		name=task_name, 
		input_module=resnet_model,
		head_module=resnet_model.fc,
		slice_head_type=None
	)
	# task1 = MultiClassificationTask(
	# 	name=task_name + ':BASE', 
	# 	input_module=resnet_model,
	# 	head_module=resnet_model.fc
	# )
	tasks = [task0]
	
	# if task_config['overfit_on_slice'] != None:
	# 	slice_names = slice_names + [int(task_config['overfit_on_slice'])]
	# else:
	# 	slice_names = slice_names + ['BASE']



	#+1 for Base


	if task_config['active_slice_heads'].get('shared_pred'):
		shared_pred_head_module = copy.deepcopy(task0.head_module)

	payloads = []
	splits = ["train", "valid", "test"]
	splits_shuffle = [True, False, False]
	slices_names = {}
	#Create Payloads
	for i in range(3):
		payload_name = f"Payload{i}_{splits[i]}"
		dataset = get_mapillary_dataset(ROOT_DIR, 
			binary_category = 'human--person',
			split='val' if splits[i] == 'valid' else splits[i])
		data_loader = MmtlDataLoader(dataset, batch_size=task_config['batch_size'], shuffle=splits_shuffle[i])
                slice_names = list(dataset.slices.keys()) + ['BASE']
                label_to_tasks = {'labelset_gold' : task_name}
                for attr_id in slices_names:

			if task_config['overfit_on_slice'] != None and attr_id == int(task_config['overfit_on_slice']):
				s = task_config['overfit_on_slice']
				slice_labelset_name = f"labelset:{s}:pred"
				slice_task_name = f"{task_name}:{s}:pred"
				labels_to_tasks[slice_labelset_name] = slice_task_name
			
			###For pred slice head type
			if task_config['active_slice_heads'].get('pred'):
				slice_labelset_name = f"labelset:{attr_id}:pred"
				slice_task_name = f"{task_name}:{attr_id}:pred"
				labels_to_tasks[slice_labelset_name] = slice_task_name

			###for shared pred head type
			if task_config['active_slice_heads'].get('shared_pred'):
				slice_labelset_name = f"labelset:{attr_id}:shared_pred"
				slice_task_name = f"{task_name}:{attr_id}:shared_pred"
				labels_to_tasks[slice_labelset_name] = slice_task_name


			###For ind slice head type
			if task_config['active_slice_heads'].get('ind'):
				mask[mask == 0] = 2 #to follow Metal convention
				slice_labelset_name = f"labelset:{attr_id}:ind"
				slice_task_name = f"{task_name}:{attr_id}:ind"
				labels_to_tasks[slice_labelset_name] = slice_task_name

		
		payload = Payload(payload_name, data_loader, labels_to_tasks, splits[i])

		if task_config['overfit_on_slice'] != None:
			payload.remap_labelsets({'labelset:{}:pred'.format(task_config['overfit_on_slice']) : task_name }, default_none=True)

		payloads.append(payload)

	loss_multiplier =  1.0 / (2 * (len(slice_names))) 
	#generate slice tasks
	for attr_id in slice_names:

		if task_config['active_slice_heads'].get('pred'):
			slice_task_name = f"{task_name}:{attr_id}:pred"
			slice_task = create_slice_task(task0, 
										   slice_task_name, 
										   slice_head_type='pred',
										   loss_multiplier=loss_multiplier,
										  )
			tasks.append(slice_task)


		###For ind slice head type
		if task_config['active_slice_heads'].get('ind'):
			slice_task_name = f"{task_name}:{attr_id}:ind"
			slice_task = create_slice_task(task0, 
										   slice_task_name, 
										   slice_head_type='ind',
										   loss_multiplier=loss_multiplier,
										  )
			tasks.append(slice_task)

		###For shared pred slice head type
		if task_config['active_slice_heads'].get('shared_pred'):
			slice_task = copy.copy(task0)
			slice_task.name = f"{task_name}:{attr_id}:shared_pred"
			slice_task.slice_head_type = 'pred'
			slice_task.head_module = shared_pred_head_module
			tasks.append(slice_task)

	return tasks, payloads
