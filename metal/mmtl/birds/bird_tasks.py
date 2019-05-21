
import sys
import metal
import os
import pickle
from resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F

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



def get_slice_funcs(slice_names, attrs_dict):
	slice_fns = {}
	
	for attr_id in slice_names:
		f = lambda x: 1 if x in attrs_dict[attr_id] else 0
		def slice_fn(x):
			in_attr = map(f, x)
			m = np.array(in_attr)
			return np.reshape(m, (m.shape[0], 1))
		
		slice_fns[str(attr_id)] = slice_fn

	return slice_fns


'''
[slice_names] is a list of attribute ids
'''
def create_birds_tasks_payloads(slice_names, X_splits, Y_splits, image_id_splits, attrs_dict, **task_config):
	set_seed(task_config['seed'])
	
	NUM_CLASSES = 200
	resnet_model = resnet50(use_as_feature_extractor=True, pretrained=True).float().cuda()
	resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
	

	task_name = 'BirdClassificationTask'
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
	slice_names = slice_names + ['BASE']
	if task_config['overfit_on_slice'] != None:
		slice_names.append(int(task_config['overfit_on_slice']))


	

	loss_multiplier =  1.0 / (2 * (len(slice_names))) #+1 for Base


	if task_config['active_slice_heads'].get('shared_pred'):
		shared_pred_head_module = copy.deepcopy(task0.head_module)

	
	#generate slice tasks
	for attr_id in slice_names:
		###For pred slice head type
		# if task_config['overfit_on_slice'] != None and attr_id == int(task_config['overfit_on_slice']):
		# 	slice_task_name = f"{task_name}:{attr_id}:pred"
		# 	slice_task = create_slice_task(task0, 
		# 								   slice_task_name, 
		# 								   slice_head_type='pred',
		# 								   loss_multiplier=loss_multiplier,
		# 								  )
		# 	tasks.append(slice_task)
		#pdb.set_trace()


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

	payloads = []
	splits = ["train", "valid", "test"]
	splits_shuffle = [True, False, False]
	train_image_ids, valid_image_ids, test_image_ids = image_id_splits
	labels_to_tasks = {"labelset_gold": task_name}
	#Create Payloads
	for i in range(3):
		payload_name = f"Payload{i}_{splits[i]}"
		X_dict = {'data': X_splits[i]}
		Y_dict = {'labelset_gold': Y_splits[i]}

		if splits[i] == 'train':
			image_ids = train_image_ids
		elif splits[i] == 'valid':
			image_ids = valid_image_ids
		else:
			image_ids = test_image_ids

		# if task_config['active_slice_heads']['ind']:
		# 	slice_labelset_name = f"labelset:BASE:ind"
		# 	slice_task_name = f"{task_name}:BASE:ind"
		# 	Y_dict[slice_labelset_name] = torch.ones(Y_splits[i].shape)
		# 	labels_to_tasks[slice_labelset_name] = slice_task_name
		
		for attr_id in slice_names:
			if attr_id == 'BASE':
				mask = torch.ones(Y_splits[i].shape).long()
			else:
				f = lambda x: 1 if x in attrs_dict[attr_id] else 0
				mask = list(map(f, image_ids.tolist()))
				mask = torch.tensor(mask)

			if task_config['overfit_on_slice'] != None and attr_id == int(task_config['overfit_on_slice']):
				s = task_config['overfit_on_slice']
				slice_labelset_name = f"labelset:{s}:pred"
				slice_task_name = f"{task_name}:{s}:pred"
				Y_dict[slice_labelset_name] = mask * Y_splits[i]
				labels_to_tasks[slice_labelset_name] = slice_task_name
			
			###For pred slice head type
			if task_config['active_slice_heads'].get('pred'):
				slice_labelset_name = f"labelset:{attr_id}:pred"
				slice_task_name = f"{task_name}:{attr_id}:pred"
				Y_dict[slice_labelset_name] = mask * Y_splits[i]
				labels_to_tasks[slice_labelset_name] = slice_task_name

			###for shared pred head type
			if task_config['active_slice_heads'].get('shared_pred'):
				slice_labelset_name = f"labelset:{attr_id}:shared_pred"
				slice_task_name = f"{task_name}:{attr_id}:shared_pred"
				Y_dict[slice_labelset_name] = mask * Y_splits[i]
				labels_to_tasks[slice_labelset_name] = slice_task_name


			###For ind slice head type
			if task_config['active_slice_heads'].get('ind'):
				mask[mask == 0] = 2 #to follow Metal convention
				slice_labelset_name = f"labelset:{attr_id}:ind"
				slice_task_name = f"{task_name}:{attr_id}:ind"
				Y_dict[slice_labelset_name] = mask 
				labels_to_tasks[slice_labelset_name] = slice_task_name

		dataset = MmtlDataset(X_dict, Y_dict)
		data_loader = MmtlDataLoader(dataset, batch_size=task_config['batch_size'], shuffle=splits_shuffle[i])
		payload = Payload(payload_name, data_loader, labels_to_tasks, splits[i])

		if task_config['overfit_on_slice'] != None:
			payload.remap_labelsets({'labelset:{}:pred'.format(task_config['overfit_on_slice']) : task_name }, default_none=True)

		payloads.append(payload)
	
	return tasks, payloads