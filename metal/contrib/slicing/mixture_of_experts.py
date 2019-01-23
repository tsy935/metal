import numpy as np
import torch
from metal.end_model import EndModel
from metal.end_model.em_defaults import em_default_config
from metal.classifier import Classifier
from metal.utils import hard_to_soft, recursive_merge_dicts
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.contrib.slicing.experiment_utils import slice_mask_from_targeting_lfs_idx
from metal.contrib.slicing.online_dp import MLPModule
import torch.nn as nn
import torch.nn.functional as F

def trainMoE(model_config, Xs, Ls, Ys, dataset_class=None, verbose=False, train_kwargs={}):
    X_train, X_dev = tuple(Xs)
    Y_train, Y_dev = tuple(Ys)
    L_train, L_dev = tuple(Ls)
    
    base_model_class = model_config["base_model_class"]
    base_model_init_kwargs = model_config["base_model_init_kwargs"]
#    input_module_class = model_config["input_module_class"]
#    input_module_init_kwargs = model_config["input_module_init_kwargs"]


    trained_models = {}
    
    # treat each LF as an expert
    m = L_train.shape[1]
    for lf_idx in range(m):
        slice_mask_train = slice_mask_from_targeting_lfs_idx(L_train, [lf_idx])
        slice_mask_dev = slice_mask_from_targeting_lfs_idx(L_dev, [lf_idx])
        slice_name = f"slice_{lf_idx}_expert"
        
        print (f"{'-'*10}Training {slice_name}{'-'*10}")
        # initialize expert end model
#        input_module = input_module_class(**input_module_init_kwargs)
        base_model = base_model_class(verbose=verbose, **base_model_init_kwargs)

        # update labels to target slices
        Y_expert_train = Y_train.copy()
        Y_expert_train[slice_mask_train] = 1
        Y_expert_train[np.logical_not(slice_mask_train)] = 2

        Y_expert_dev = Y_dev.copy()
        Y_expert_dev[slice_mask_dev] = 1
        Y_expert_dev[np.logical_not(slice_mask_dev)] = 2

        # create slice-specific data loaders      
        if dataset_class:
            train = dataset_class(X_train, Y_train, slice_mask=slice_mask_train)
            dev = dataset_class(X_dev, Y_dev, slice_mask=slice_mask_dev, training=False)
        else: 
            train = (X_train, Y_train)
            dev = (X_dev, Y_dev)

        # train expert
        base_model.train_model(train, dev_data=dev, n_epochs=10, disable_prog_bar=True, verbose=verbose)
        score = base_model.score(dev, verbose=verbose)
        print (f"Dev Score on L{lf_idx} examples:", score)
        trained_models[slice_name] = base_model

    # train mixture of experts model
    moe = MoEModel(trained_models, d=X_train.shape[1])
    if dataset_class:
        train = dataset_class(X_train, Y_train)
        dev = dataset_class(X_dev, Y_dev, training=False)
    else:
        train = (X_train, Y_train)
        dev = (X_dev, Y_dev)
        

    moe.train_model(train, dev_data=dev, **train_kwargs)
    return moe

class MoEModel(Classifier):
    def __init__(self, pretrained_experts, d, k=2, gating_dim=10, **kwargs):
        """
        Args:
            pretrained_experts: dict mapping expert_name to pretrained model
            d: dimension of data
            k: number of output classes
        """
        config = recursive_merge_dicts(
            em_default_config, kwargs, misses="insert"
        )
        super().__init__(k=k, config=config)
        
        self.experts = pretrained_experts
        self.d = d
        self.k = k
        
        # gating network
        num_experts = len(pretrained_experts)
        
        # output of input_data --> weights for each expert
        self.gating = nn.Sequential(
            nn.Linear(d, gating_dim),
            nn.ReLU(),
            nn.Linear(gating_dim, num_experts),
            nn.Softmax()
        )
        
        self.criteria = SoftCrossEntropyLoss(reduction='sum')
        
        # freeze all weights        
        for model_name, model in self.experts.items():
            for param in model.parameters():
                param.requires_grad = False

    def _expert_forward(self, X):
        # stacked_preds [num_experts, batch_size, k]
        stacked_preds = torch.stack([model(X) for model in self.experts.values()]).contiguous()
        # preds [batch_size, num_experts, k]
        preds = torch.transpose(stacked_preds, 0, 1)
        return preds
        
    def weighted_experts(self, X):
         # [batch_size, num_experts, k]
        expert_logits = self._expert_forward(X)
        
        # [batch_size, 1, num_experts]
        gating_weights = self.gating(X).unsqueeze(1)

        # [batch_size, num_experts, k]
        weighted_logits = torch.bmm(gating_weights, expert_logits).squeeze(1)
        return weighted_logits
    
    def _loss(self, X, Y):
        weighted_logits = self.weighted_experts(X)
        loss = self.criteria(weighted_logits, self._preprocess_Y(Y, self.k))
        return loss
    
    def _get_loss_fn(self):
        return self._loss
    
    def predict_proba(self, X):
        return F.softmax(self.weighted_experts(X)).detach().numpy()

    
    def train_model(self, train_data, dev_data=None, log_writer=None, **kwargs):
        self.config = recursive_merge_dicts(self.config, kwargs)

        # If train_data is provided as a tuple (X, Y), we can make sure Y is in
        # the correct format
        # NOTE: Better handling for if train_data is Dataset or DataLoader...?
        if isinstance(train_data, (tuple, list)):
            X, Y = train_data
            Y = self._preprocess_Y(
                self._to_torch(Y, dtype=torch.FloatTensor), self.k
            )
            train_data = (X, Y)

        # Convert input data to data loaders
        train_loader = self._create_data_loader(train_data, shuffle=True)
        dev_loader = self._create_data_loader(dev_data, shuffle=False)

        # Create loss function
        loss_fn = self._get_loss_fn()

        # Execute training procedure
        self._train_model(
            train_loader, loss_fn, dev_data=dev_loader, log_writer=log_writer
        )
    
    def _preprocess_Y(self, Y, k):
        """Convert Y to soft labels if necessary"""
        Y = Y.clone()

        # If hard labels, convert to soft labels
        if Y.dim() == 1 or Y.shape[1] == 1:
            Y = hard_to_soft(Y.long(), k=k)
        return Y
