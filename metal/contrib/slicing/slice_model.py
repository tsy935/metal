import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from metal.classifier import Classifier
from metal.end_model.end_model import EndModel
from metal.utils import SlicingDataset, recursive_merge_dicts

slice_defaults = {
    "L_weight": 0.1,
    "Y_off_weight": 0.1,
    "vanilla": False,
    "Y_off_weight": 0.1,
    "online_head": False,
    "indicator": True,
    "mask_abstains": False,
}


class SliceMaster(EndModel):
    """A model which makes an EndModel slice-aware

    Args:
        base_model: an instantiated EndModel (to be copied and reinitialized)
        m: the number of labeling functions
            if unipolarizing, this is the number post-unipolarizing
        L_weight: a float in [0,1]; how much the L_loss matters
            0: ignore the L head (it is a vanilla EndModel)
            1: ignore the Y head (focus only on recreating L)
        Y_off_weight: a float in [0,1]; how much the Y_off_loss matters
            if online_head=False, this is overriden to be 1 - L_weight
        online_head: (bool) if True, predictions are made by an online Y_head
            that accepts the representation of both L_head and offline Y_head
        indicator: (bool) Denotes whether L heads predict indicators or labels
            True: L heads predict indicators of labels (abstain = 0)
            False: L heads predict labels directly (abstain = 0.5)
        vanilla: (bool)
            if True, override all other options and make this a vanilla EndModel
                (L_head and Y_offline_head won't even be instantiated)
        mask_abstains: (bool) if True, mask the loss from abstains in L_head
            Must be False if indicator=True

    Old settings:
        EndModel (DP): vanilla=True (automatic: online_head=False, L_weight=0)
            Existence of extra params may also affect l2 until they drop to 0
            extra params may come from L_head or Y_head_on.
            Also, plan EndModel is k=2, new model is k=1.
        SliceDP:       online_head=False, indicator=True
        SliceDPOnline: online_head=True,  indicator=True
        SliceHat:      online_head=False, indicator=False
        SliceOnline:   online_head=True,  indicator=False
    """

    def __init__(self, base_model, m, **kwargs):
        kwargs["seed"] = kwargs.get("seed", None)  # Override base_model's seed
        config = recursive_merge_dicts(
            base_model.config, slice_defaults, misses="insert"
        )
        config = recursive_merge_dicts(config, kwargs)

        if config["vanilla"]:
            print("Overriding options to create vanilla EndModel.")
            config["online_head"] = False
            config["L_weight"] = 0.0
        if config["indicator"] and config["mask_abstains"]:
            raise Exception("If indicator=True, mask_abstains must be False")

        k = base_model.network[-1].out_features
        if k != 2:
            raise Exception("SliceMaster is currently only valid for k=2.")

        # Leap frog EndModel initialization (`model` is already initialized)
        Classifier.__init__(self, k=k, config=config)
        self.m = m
        self.online_head = self.config["online_head"]
        self.indicator = self.config["indicator"]
        self.vanilla = self.config["vanilla"]
        self.mask_abstains = self.config["mask_abstains"]
        self.build_model(base_model)

        # Precalculate loss fractions
        self.L_weight = config["L_weight"]
        if self.online_head:
            self.Y_off_weight = config["Y_off_weight"]
            self.Y_on_weight = 1 - self.L_weight - self.Y_off_weight
        else:
            self.Y_off_weight = 1 - self.L_weight
            self.Y_on_weight = 0

        # Show network
        if self.config["verbose"]:
            print(self)
            print()

    def build_model(self, base_model):
        Y_head_off = base_model.network[-1]
        neck_dim = Y_head_off.in_features

        # Reset model body
        self.body = copy.deepcopy(base_model.network[:-1])
        if self.config["verbose"]:
            print("Resetting base model parameters")
        self.body.apply(reset_parameters)

        # No bias on L-head; we only want the weights
        if not self.vanilla:
            self.L_head = nn.Linear(neck_dim, self.m, bias=False)
            self.L_criteria = nn.BCEWithLogitsLoss(reduction="none")

        # WARNING: breaking MeTaL convention and using 1-dim output when k=2
        self.Y_head_off = nn.Linear(neck_dim, 1)
        self.Y_criteria = nn.BCEWithLogitsLoss(reduction="mean")

        # Add online Y head
        if self.online_head and not self.vanilla:
            self.Y_head_on = nn.Linear(neck_dim, 1)

    def _get_loss_fn(self):
        if self.vanilla:
            return self._vanilla_loss
        else:
            return self._loss

    def _vanilla_loss(self, L, X, Y_s, _):
        neck = self.body(X)
        Y_off_logits = self.forward_Y_off(neck)
        return self.Y_criteria(Y_off_logits.squeeze(), Y_s[:, 0].float())

    def _loss(self, L, X, Y_s, _):
        """Returns the average loss per example"""
        neck = self.body(X)

        if self.mask_abstains:
            abstains = L == 0

        # TODO: Do not change L types every time the loss function is called!
        # The issue is that MeTaL uses categorical {0,1,2} labels throughout
        # and BCEWithLogitsLoss expects onezero {0.5,1,0} labels
        L = L.clone()
        if self.indicator:
            L[L == 2] = 1
        else:
            L[L == 0] = 0.5  # Abstains are ambivalent (0 logit)
            L[L == 2] = 0  # 2s are negative class

        L_logits = self.forward_L(neck)
        L_loss = self.L_criteria(L_logits, L.float())
        if self.mask_abstains:
            L_loss = L_loss.masked_fill(abstains, 0)

        # Get average L loss per example per lf
        L_loss = torch.mean(L_loss.sum(dim=1), dim=0) / self.m

        # Execute Y offline head
        Y_off_logits = self.forward_Y_off(neck, L_logits)
        Y_off_loss = self.Y_criteria(Y_off_logits.squeeze(), Y_s[:, 0].float())

        # Execute Y online head
        if self.online_head:
            Y_on_logits = self.forward_Y_on(neck, L_logits)
            Y_on_loss = self.Y_criteria(
                Y_on_logits.squeeze(), Y_s[:, 0].float()
            )
        else:
            Y_on_loss = 0

        # Combine losses
        loss = (
            self.L_weight * L_loss
            + self.Y_off_weight * Y_off_loss
            + self.Y_on_weight * Y_on_loss
        )
        return loss

    def update_neck(self, neck, L_logits):
        # A is the [batch_size, m] Tensor representing the amount of
        # attention to pay to each head (based on each ones' confidence)
        A = F.softmax(abs(L_logits), dim=1)
        # W is the [r, m] linear mapping that transforms the neck into
        # logits for the L heads
        W = self.L_head.weight
        # R is the [r, 1] neck (representation vector) that would
        # normally go into the Y head
        R = neck
        # We reweight the linear mappings W by the attention scores A
        # and use this to create a reweighted representation S
        S = (A @ W) * R
        # Add this to R proportional to L_weight
        # NOTE: Reweighting S before combining with R is new
        # It is an attempt to allow a smooth transition to increased attention
        # on the L_head.
        # We actually have hard param sharing from loss and soft param sharing
        # from this mixing, and these need not have same weight coefficient.
        return R + (S * self.L_weight)

    def forward_L(self, neck):
        """Returns the logits of the L_head"""
        return self.L_head(neck)

    def forward_Y_off(self, neck, L_logits=None):
        """Returns the logits of the offline Y_head"""
        if not self.online_head and not self.vanilla:
            if L_logits is None:
                L_logits = self.forward_L(neck)
            neck = self.update_neck(neck, L_logits)
        return self.Y_head_off(neck)

    def forward_Y_on(self, neck, L_logits=None):
        """Returns the logits of the online Y_head"""
        if L_logits is None:
            L_logits = self.forward_L(neck)
        neck = self.update_neck(neck, L_logits)
        return self.Y_head_on(neck)

    def _get_predictions(self, data, **kwargs):
        if isinstance(data, tuple):
            L, X, Y, Z = data
            data_loader = DataLoader(SlicingDataset(X, Y))
        elif isinstance(data, Dataset):
            # eval on X, Y
            data_loader = DataLoader(data.data[1], data.data[2])
        elif isinstance(data, DataLoader):
            data_loader = DataLoader(
                SlicingDataset(data.dataset.data[1], data.dataset.data[2])
            )
        else:
            raise NotImplementedError(
                f"Unrecognized type for data: {type(data)}"
            )

        return super()._get_predictions(data_loader, **kwargs)

    @torch.no_grad()
    def predict_L_proba(self, X):
        """A convenience function that predicts L probabilities"""
        neck = self.body(X)
        return torch.sigmoid(self.L_head(neck))

    @torch.no_grad()
    def predict_Y_off_proba(self, X):
        """A convenience function that predicts Y offline probabilities"""
        neck = self.body(X)
        probs = torch.sigmoid(self.forward_Y_off(neck)).data.cpu().numpy()
        return np.hstack((probs, 1 - probs))

    @torch.no_grad()
    def predict_Y_on_proba(self, X):
        """A convenience function that predicts Y online probabilities"""
        assert self.online_head
        neck = self.body(X)
        probs = torch.sigmoid(self.forward_Y_on(neck)).data.cpu().numpy()
        return np.hstack((probs, 1 - probs))

    @torch.no_grad()
    def predict_all_proba(self, X):
        """A convenience function that predicts all heads probs at once"""
        neck = self.body(X)
        L_logits = self.L_head(neck)
        L_probs = torch.sigmoid(L_logits)
        Y_off_probs = (
            torch.sigmoid(self.forward_Y_off(neck, L_logits)).data.cpu().numpy()
        )
        if self.online_head:
            Y_on_probs = (
                torch.sigmoid(self.forward_Y_on(neck, L_logits))
                .data.cpu()
                .numpy()
            )
        else:
            Y_on_probs = None
        return L_probs, Y_off_probs, Y_on_probs

    @torch.no_grad()
    def predict_proba(self, X):
        if self.online_head:
            return self.predict_Y_on_proba(X)
        else:
            return self.predict_Y_off_proba(X)


def reset_parameters(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass
