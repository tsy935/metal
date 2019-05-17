
import sys
import metal
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.mmtl.payload import Payload
from metal.mmtl.data import MmtlDataLoader, MmtlDataset
from pprint import pprint
from metal.mmtl.slicing.tasks import *
from metal.mmtl.metal_model import MetalModel 
from metal.mmtl.trainer import MultitaskTrainer



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
def create_birds_tasks_payloads(slice_names, ind_head, pred_head, X_splits, Y_splits, image_id_splits, attrs_dict, model):
	resnet_model = model
	task_name = 'BirdClassificationTask'
	task0 = MultiClassificationTask(
		name=task_name, 
		input_module=resnet_model,
		head_module=resnet_model.fc
	)
	task1 = MultiClassificationTask(
		name=task_name + ':BASE', 
		input_module=resnet_model,
		head_module=resnet_model.fc
	)
	tasks = [task0]
	loss_multiplier =  1.0 / (2 * (len(slice_names) +1 )) #+1 for Base

	slice_task_name = f"{task_name}:BASE:ind"
	slice_task = create_slice_task(task0, 
									   slice_task_name, 
									   slice_head_type='ind',
									   loss_multiplier=loss_multiplier,
									   classification_task=MultiClassificationTask,
									  )
	tasks.append(slice_task)
	
	
	


	#generate slice tasks
	for attr_id in slice_names:
		###For pred slice head type
		if pred_head:
			slice_task_name = f"{task_name}:{attr_id}:pred"
			slice_task = create_slice_task(task0, 
										   slice_task_name, 
										   slice_head_type='pred',
										   loss_multiplier=loss_multiplier,
										   classification_task=MultiClassificationTask,
										  )
			tasks.append(slice_task)


		###For ind slice head type
		if ind_head:
			slice_task_name = f"{task_name}:{attr_id}:ind"
			slice_task = create_slice_task(task0, 
										   slice_task_name, 
										   slice_head_type='ind',
										   loss_multiplier=loss_multiplier,
										   classification_task=MultiClassificationTask
										  )
			tasks.append(slice_task)

	payloads = []
	splits = ["train", "valid", "test"]
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

		slice_labelset_name = f"labelset:BASE:ind"
		slice_task_name = f"{task_name}:BASE:ind"
		Y_dict[slice_labelset_name] = torch.ones(Y_splits[i].shape)
		labels_to_tasks[slice_labelset_name] = slice_task_name
		

		for attr_id in slice_names:
			f = lambda x: 1 if x in attrs_dict[attr_id] else 0
			mask = list(map(f, image_ids.tolist()))
			mask = torch.tensor(mask)
			
			###For pred slice head type
			if pred_head:
				slice_labelset_name = f"labelset:{attr_id}:pred"
				slice_task_name = f"{task_name}:{attr_id}:pred"
				Y_dict[slice_labelset_name] = mask * Y_splits[i]
				labels_to_tasks[slice_labelset_name] = slice_task_name

			###For ind slice head type
			if ind_head:
				mask[mask == 0] = 2 #to follow Metal convention
				slice_labelset_name = f"labelset:{attr_id}:ind"
				slice_task_name = f"{task_name}:{attr_id}:ind"
				Y_dict[slice_labelset_name] = mask 
				labels_to_tasks[slice_labelset_name] = slice_task_name

		dataset = MmtlDataset(X_dict, Y_dict)
		data_loader = MmtlDataLoader(dataset, batch_size=32)
		payload = Payload(payload_name, data_loader, labels_to_tasks, splits[i])
		payloads.append(payload)
	
	return tasks, payloads