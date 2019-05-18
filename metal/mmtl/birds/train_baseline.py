import sys, os
sys.path.append('../../../../metal')
os.environ['METALHOME'] = '/dfs/scratch1/saelig/slicing/metal'
import metal
import torch
import torch.nn as nn
from metal.mmtl.slicing.tasks import MultiClassificationTask
from metal.mmtl.metal_model import MetalModel 
from resnet import *
from metal.mmtl.trainer import MultitaskTrainer
from metal.mmtl.payload import Payload
from pprint import pprint

opj = os.path.join
HOME_DIR = '/dfs/scratch1/saelig/slicing/'
DATASET_DIR = opj(HOME_DIR,'CUB_200_2011')
IMAGES_DIR = opj(DATASET_DIR, 'images')
TENSORS_DIR = opj(HOME_DIR, 'birds_data')
MODELS_DIR = opj(HOME_DIR, 'birds_models')

if __name__ == '__main__':

	X_train = torch.load(opj(TENSORS_DIR,'X_train.pt'))
	X_valid = torch.load(opj(TENSORS_DIR,'X_valid.pt'))
	X_test = torch.load(opj(TENSORS_DIR,'X_test.pt'))
	Y_train = torch.load(opj(TENSORS_DIR,'Y_train.pt'))
	Y_valid = torch.load(opj(TENSORS_DIR,'Y_valid.pt'))
	Y_test = torch.load(opj(TENSORS_DIR,'Y_test.pt'))

	NUM_CLASSES = 200
	resnet_model = resnet50(use_as_feature_extractor=True, pretrained=True).float().cuda()
	resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
	task0 = MultiClassificationTask(
	    name='BirdClassificationTask', 
	    input_module=resnet_model,
	    head_module=resnet_model.fc
	)
	tasks = [task0]
	model = MetalModel(tasks, verbose=False)

	payloads = []
	splits = ["train", "valid", "test"]
	X_splits = X_train, X_valid, X_test
	Y_splits = Y_train, Y_valid, Y_test

	for i in range(3):
	    payload_name = f"Payload{i}_{splits[i]}"
	    task_name = task0.name
	    #print(X_splits[i].shape)
	    if splits[i] == 'train': #shuffle while training
	        payload = Payload.from_tensors(payload_name, {'data': X_splits[i]}, {'labels' : Y_splits[i]}, task_name, splits[i], shuffle=True, batch_size=16)
	    else:
	        payload = Payload.from_tensors(payload_name, {'data': X_splits[i]}, {'labels' : Y_splits[i]}, task_name, splits[i], batch_size=16)
	    #payload = Payload.from_tensors(payload_name, X_splits[i], Y_splits[i], task_name, splits[i], batch_size=32)
	    payloads.append(payload)


	trainer = MultitaskTrainer()
	scores = trainer.train_model(
	    model, 
	    payloads, 
	    n_epochs=100, 
	    log_every=2,
	    lr=0.01,
	    progress_bar=False,
	    lr_scheduler='reduce_on_plateau',
	    patience=10,
	    factor=0.1,
	    checkpoint_every=2,
	    checkpoint_metric='BirdClassificationTask/Payload1_valid/labels/accuracy',
	    checkpoint_metric_mode='max',
	    # log_dir=f"{os.environ['METALHOME']}/logs/",
	    # checkpoint_dir=f"{os.environ['METALHOME']}/checkpoints/baseline",
	)