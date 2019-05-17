import sys, os
sys.path.append('../../../../metal')
os.environ['METALHOME'] = '/dfs/scratch1/saelig/slicing/metal/'

import pickle
import torch
from pprint import pprint
import metal
from bird_tasks import *

from metal.mmtl.metal_model import MetalModel
from metal.mmtl.slicing.slice_model import SliceModel, SliceRepModel
from metal.mmtl.slicing.moe_model import MoEModel
from resnet import *





opj = os.path.join
HOME_DIR = '/dfs/scratch1/saelig/slicing/'
DATASET_DIR = opj(HOME_DIR,'CUB_200_2011')
IMAGES_DIR = opj(DATASET_DIR, 'images')
TENSORS_DIR = opj(HOME_DIR, 'birds_data')
MODELS_DIR = opj(HOME_DIR, 'birds_models')






if __name__ == '__main__':
	print('Loading data...')
	train_image_ids = torch.load(opj(TENSORS_DIR,'train_image_ids.pt'))
	valid_image_ids = torch.load(opj(TENSORS_DIR,'valid_image_ids.pt'))
	test_image_ids = torch.load(opj(TENSORS_DIR,'test_image_ids.pt'))
	X_train = torch.load(opj(TENSORS_DIR,'X_train.pt'))
	X_valid = torch.load(opj(TENSORS_DIR,'X_valid.pt'))
	X_test = torch.load(opj(TENSORS_DIR,'X_test.pt'))
	Y_train = torch.load(opj(TENSORS_DIR,'Y_train.pt'))
	Y_valid = torch.load(opj(TENSORS_DIR,'Y_valid.pt'))
	Y_test = torch.load(opj(TENSORS_DIR,'Y_test.pt'))
	attrs_dict = pickle.load(open(opj(TENSORS_DIR, 'attrs_dict.pkl'),'rb'))
	print('Done')

	image_id_splits = train_image_ids, valid_image_ids, test_image_ids
	X_splits = X_train, X_valid, X_test
	Y_splits = Y_train, Y_valid, Y_test

	resnet_model = resnet18(num_classes=200, use_as_feature_extractor=True).float().cuda()
	slice_names = [23, 24]

	ind_head = True; pred_head = False
	tasks, payloads = create_birds_tasks_payloads(slice_names, ind_head, pred_head, X_splits, Y_splits, image_id_splits, attrs_dict, resnet_model)

	print('Tasks: ')
	pprint(tasks)
	print('\n\n Payloads: ')
	pprint(payloads)

	

	slice_train_attrs = ['23', '24']
	slice_train_funcs = get_slice_funcs(slice_names, attrs_dict)
	identity_fn = lambda x: np.ones(x.shape[0], dtype=np.bool)
	slice_train_funcs['BASE'] = identity_fn
	slice_loss_multiplier = 1.0 / (2*(len(slice_train_funcs) + 1))
	slice_weights = {attr:slice_loss_multiplier for attr in slice_train_attrs}
	slice_weights_w_base = dict(slice_weights)
	slice_weights_w_base['BASE'] = slice_loss_multiplier

	model_configs = {
	    'soft_param_rep': {
	        'slice_funcs': slice_train_funcs,
	        'create_ind': True,
	        'create_preds': False,
	        'model_class': SliceRepModel,
	        'slice_weights' : slice_weights_w_base,
	#         'slice_weights': {
	#             'BASE': slice_loss_multiplier,
	#             'slice_1': slice_loss_multiplier, 'slice_2': slice_loss_multiplier
	#         },
	        'h_dim': 2
	    }
	}

	for model_name, config in model_configs.items():
	    # pretrained_input_module = resnet_model.input_modules['BirdClassificationTask'].module.module
	    # pretrained_head_module = resnet_model.head_modules['BirdClassificationTask'].module.module
	    

	    print(f"{'='*10}Initializing + Training {model_name}{'='*10}")
	    slice_funcs = config['slice_funcs']
	    model_class = config['model_class']
	    slice_weights = config.get("slice_weights", {})
	    create_ind = config.get("create_ind", True)
	    create_preds = config.get("create_preds", True)
	    h_dim = config.get("h_dim", None)
	    # just the one task
	    #get payloads from above
	        
	    if model_name == 'moe':
	        # train for same total num epochs
	        expert_train_kwargs = copy.deepcopy(train_kwargs)
	        expert_train_kwargs['n_epochs'] = int(train_kwargs['n_epochs'] / (len(all_slice_funcs) + 1))
	        experts = train_slice_experts(
	            uid_lists, Xs, Ys, MetalModel, all_slice_funcs, **expert_train_kwargs
	        )
	        model = model_class(tasks, experts, verbose=False, seed=seed)
	        trainer = MultitaskTrainer(seed=seed)
	        metrics_dict = trainer.train_model(model, payloads, **expert_train_kwargs)
	    else:
	        model = model_class(tasks, h_dim=h_dim, verbose=True)
	        trainer = MultitaskTrainer()
	        metrics_dict = trainer.train_model(model, payloads, **train_kwargs)
	    print(metrics_dict) 
	    trained_models[model_name] = model























