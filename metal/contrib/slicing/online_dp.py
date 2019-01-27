import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.classifier import Classifier
from metal.end_model.em_defaults import em_default_config
from metal.end_model.end_model import EndModel
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.utils import recursive_merge_dicts


class LinearModule(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.input_layer(x)


class MLPModule(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dims=[], bias=True):
        super().__init__()

        # Create layers
        dims = [input_dim] + middle_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i + 1 < len(dims):
                layers.append(nn.ReLU())

        self.input_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.input_layer(x)


class SliceDPModel(EndModel):
    """
    Args:
        - input_module: (nn.Module) a module that converts the user-provided
            model inputs to torch.Tensors. Defaults to IdentityModule.
        - r: Intermediate representation dimension
        - m: number of labeling functions AKA size of L_head
        - reweight: Whether to use reweighting of representation for Y_head
        - L_weights: The m-dim vector of weights to use for the LF-head
                loss weighting; defaults to all 1's.
        - slice-weight: Factor to multiply loss for L_heads in joint loss with
                Y_head loss
        - middle_modules: (nn.Module) a list of modules to execute between the
            input_module and task head. Defaults to nn.Linear.
        - head_module: (nn.Module) a module to execute right before the final
            softmax that outputs a prediction for the task.
    """

    def __init__(
        self,
        input_module,
        r,
        m,
        reweight=False,
        L_weights=None,
        slice_weight=0.5,
        middle_modules=None,
        verbose=True,
        **kwargs
    ):
        self.r = r
        self.m = m  # number of labeling sources
        self.reweight = reweight
        self.output_dim = 2  # NOTE: Fixed for binary setting
        self.slice_weight = slice_weight

        # No bias -- only learn weights for L_head
        head_module = nn.Linear(self.r, self.m, bias=False)

        # default to removing nonlinearities from input_layer output
        input_layer_config = kwargs.get(
            "input_layer_config",
            {
                "input_relu": False,
                "input_batchnorm": False,
                "input_dropout": 0.0,
            },
        )
        kwargs["input_layer_config"] = input_layer_config

        # Initialize EndModel.
        # Note: We overwrite `self.k` which conventionally refers to the
        # number of tasks to `self.m` which is the number of LFs in our setting.
        super().__init__(
            [self.r, self.m],  # layer_out_dims
            input_module,
            middle_modules,
            head_module,
            verbose=False,  # don't print EndModel params
            **kwargs
        )

        # Set "verbose" in config
        self.update_config({"verbose": verbose})

        # Redefine loss fn
        self.criteria_L = nn.BCEWithLogitsLoss(reduce=False)
        self.criteria_Y = SoftCrossEntropyLoss(reduction="none")

        # For manually reweighting. Default to all ones.
        if L_weights is None:
            self.L_weights = torch.ones(self.m).reshape(-1, 1)
        else:
            self.L_weights = torch.from_numpy(L_weights).reshape(-1, 1)

        # Set network and L_head modules
        modules = list(self.network.children())
        self.network = nn.Sequential(*list(modules[:-1]))
        self.L_head = modules[-1]

        # Attach the "DP head" which outputs the final prediction
        y_d = 2 * self.r if self.reweight else self.r
        self.Y_head = nn.Linear(y_d, self.output_dim)

        if self.config["use_cuda"]:
            # L_weights: how much to weight the loss for each L_head
            self.L_weights = self.L_weights.cuda()

        if self.config["verbose"]:
            print("Slice Heads:")
            print("Reweighting:", self.reweight)
            print("L_weights:", self.L_weights)
            print("Slice Weight:", self.slice_weight)
            print("Input Network:", self.network)
            print("L_head:", self.L_head)
            print("Y_head:", self.Y_head)
            print("Criteria:", self.criteria_L, self.criteria_Y)

    def _loss(self, X, L, Y_weak):
        """Returns the loss consisting of summing the LF + DP head losses

        Args:
            - X: A [batch_size, d] torch Tensor
            - L: A [batch_size, m] torch Tensor with elements in {0,1,2}
            - Y_weak: A [batch_size, k] torch Tensor labels with elements [0,1] for
                each col k class
        """

        # L indicates whether LF is triggered; supports multiclass
        L[L != 0] = 1

        # LF heads loss
        loss_1 = torch.mean(
            self.criteria_L(self.forward_L(X), L) @ self.L_weights
        )

        # Mask if none of the slices are activated-- don't backprop
        dp_head_mask = (torch.sum(abs(L), dim=1) > 0).float()
        masked_Y_loss = (
            self.criteria_Y(self.forward_Y(X), Y_weak) * dp_head_mask
        )
        # print("WARNING: now summing loss i/o average")
        loss_2 = torch.sum(masked_Y_loss)

        # Compute the weighted sum of these
        loss_1 /= self.m  # normalize by number of LFs
        loss = (self.slice_weight * loss_1) + ((1 - self.slice_weight) * loss_2)
        return loss

    def _get_loss_fn(self):
        """ Override `EndModel` loss function with custom L_head + Y_head loss"""
        return self._loss

    def forward_L(self, x):
        """Returns the unnormalized predictions of the L_head layer."""
        return self.L_head(self.network(x))

    def forward_Y(self, x):
        """Returns the output of the Y head only, over re-weighted repr."""
        batchsize = x.shape[0]
        xr = self.network(x)

        # Concatenate with the LF attention-weighted representation as well
        if self.reweight:
            # A is the [batch_size, 1, m] Tensor representing the confidences
            # that the example belongs to each L_head
            A = F.softmax(self.forward_L(x)).unsqueeze(1)

            # We then project the A weighting onto the respective features of
            # the L_head layer, and add these attention-weighted features to Xr
            W = self.L_head.weight.repeat(batchsize, 1, 1)
            xr = torch.cat([xr, torch.bmm(A, W).squeeze()], 1)

        # Return the list of head outputs + DP head
        outputs = self.Y_head(xr).squeeze()
        return outputs

    def predict_proba(self, x):
        with torch.no_grad():
            return F.softmax(self.forward_Y(x)).data.cpu().numpy()


class SliceHatModel(EndModel):
    """A model which makes a base single-task EndModel slice-aware

    Args:
        base_model: an instantiated EndModel (it will be copied and reinitialized)
        m: the number of labeling functions
        slice_weight: a float in [0,1]; how much the L_loss matters
            0: there is no L head (it is a vanilla EndModel)
            1: there is no Y head (it learns only to predict Ys)
        reweight: (bool) if True, use attention to reweight the neck
    """

    def __init__(
        self, base_model, m, slice_weight=0.1, reweight=True, **kwargs
    ):
        # NOTE: rather than using em_default_config, we use base_model.config
        kwargs["slice_weight"] = slice_weight  # Add to kwargs so it merges
        # base_model has a seed, but use SliceHatModel's seed instead
        kwargs["seed"] = kwargs.get("seed", None)
        config = recursive_merge_dicts(
            base_model.config, kwargs, misses="insert"
        )
        k = base_model.network[-1].out_features
        if k != 2:
            raise Exception("SliceHatModel is currently only valid for k=2.")

        # Leap frog EndModel initialization (`model` is already initialized)
        Classifier.__init__(self, k=k, config=config)

        self.m = m
        self.slice_weight = slice_weight
        self.reweight = reweight
        self.build_model(base_model)

        # Show network
        if self.config["verbose"]:
            print(self)
            print()

    def build_model(self, base_model):
        Y_head_off = base_model.network[-1]
        neck_dim = Y_head_off.in_features
        self.body = copy.deepcopy(base_model.network[:-1])
        if self.config["verbose"]:
            print("Resetting base model parameters")
        self.body.apply(reset_parameters)

        self.has_L_head = self.slice_weight > 0
        self.has_Y_head = self.slice_weight < 1

        if self.has_L_head:
            # No bias on L-head; we only want the weights
            self.L_head = nn.Linear(neck_dim, self.m, bias=False)
            self.L_criteria = nn.BCEWithLogitsLoss(reduction="none")

        if self.has_Y_head:
            # WARNING: breaking MeTaL convention and using 1-dim output when k=2
            if self.reweight:
                if not self.has_L_head:
                    msg = "Cannot reweight neck if no L_head is present."
                    raise Exception(msg)
                # If reweighting, Y_head sees original rep and reweighted one
                self.Y_head_off = nn.Linear(2 * neck_dim, 1)
            else:
                self.Y_head_off = nn.Linear(neck_dim, 1)
            self.Y_criteria = nn.BCEWithLogitsLoss(reduction="mean")

    def _get_loss_fn(self):
        return self._loss

    def _loss(self, X, L, Y_s):
        """Returns the average loss per example"""
        # For efficiency, L would be converted to {-1,0,1} before being passed
        # into the model; for consistency in the code, we leave L in {0,1,2}
        # wherever the user deals with it.

        abstains = L == 0
        # To turn off masking:
        # abstains = torch.ones_like(L).byte()

        L = L.clone()
        L[L == 0] = 0.5  # Abstains are ambivalent (0 logit)
        L[L == 2] = 0  # 2s are negative class

        neck = self.body(X)

        if self.has_L_head:
            L_logits = self.forward_L(neck)
            # Weight loss by lf weights if applicable and mask abstains
            L_loss_masked = (
                self.L_criteria(L_logits, L.float())
                .masked_fill(abstains, 0)
                .sum(dim=1)
            )
            # Get average L loss by example per lf
            L_loss = torch.mean(L_loss_masked, dim=0) / self.m
        else:
            L_logits = None
            L_loss = 0

        if self.has_Y_head:
            Y_logits = self.forward_Y_off(X, L_logits)
            Y_loss = self.Y_criteria(Y_logits.squeeze(), Y_s[:, 0].float())
        else:
            Y_loss = 0

        loss = (self.slice_weight * L_loss) + ((1 - self.slice_weight) * Y_loss)
        return loss

    def forward_L(self, neck):
        return self.L_head(neck)

    def forward_Y_off(self, X, L_logits=None):
        """Returns the logits of the offline Y_head"""
        neck = self.body(X)
        if self.reweight:
            if L_logits is None:
                L_logits = self.L_head(neck)
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
            # Concatentate these into a single input for the Y_head
            neck = torch.cat((R, S), 1)
        return self.Y_head_off(neck)

    @torch.no_grad()
    def predict_L_proba(self, X):
        """A convenience function that predicts L probabilities"""
        return torch.sigmoid(self.L_head(self.body(X)))

    @torch.no_grad()
    def predict_proba(self, X):
        preds = torch.sigmoid(self.forward_Y_off(X)).data.cpu().numpy()
        return np.hstack((preds, 1 - preds))


class SliceOnlineModel(EndModel):
    """A model which makes an offline EndModel an online slice-aware one

    Args:
        base_model: an instantiated EndModel (it will be copied and reinitialized)
        m: the number of labeling functions
        slice_weight: a float in [0,1]; how much the L_loss matters
            0: there is no L head (it is a vanilla EndModel)
            1: there is no Y head (it learns only to predict Ys)
        reweight: (bool) if True, use attention to reweight the neck
    """

    def __init__(
        self, base_model, m, L_head_weight=0.1, Y_head_weight=0.1, **kwargs
    ):
        # NOTE: rather than using em_default_config, we use base_model.config
        # Add head weights to kwargs so they merge
        kwargs["L_head_weight"] = L_head_weight
        kwargs["Y_head_weight"] = Y_head_weight
        # base_model has a seed, but use SliceHatModel's seed instead
        kwargs["seed"] = kwargs.get("seed", None)
        config = recursive_merge_dicts(
            base_model.config, kwargs, misses="insert"
        )
        k = base_model.network[-1].out_features
        if k != 2:
            raise Exception("SliceHatModel is currently only valid for k=2.")

        # Leap frog EndModel initialization (`model` is already initialized)
        Classifier.__init__(self, k=k, config=config)

        self.m = m
        self.L_head_weight = L_head_weight
        self.Y_head_weight = Y_head_weight
        self.build_model(base_model)

        # Show network
        if self.config["verbose"]:
            print(self)
            print()

    def build_model(self, base_model):
        Y_head_off = base_model.network[-1]
        neck_dim = Y_head_off.in_features
        self.body = copy.deepcopy(base_model.network[:-1])
        if self.config["verbose"]:
            print("Resetting base model parameters")
        self.body.apply(reset_parameters)

        # No bias on L-head; we only want the weights
        self.L_head = nn.Linear(neck_dim, self.m, bias=False)
        self.L_criteria = nn.BCEWithLogitsLoss(reduction="none")

        # WARNING: breaking MeTaL convention and using 1-dim output when k=2
        self.Y_head_off = nn.Linear(neck_dim, 1)
        self.Y_criteria = nn.BCEWithLogitsLoss(reduction="mean")

        self.Y_head_on = nn.Linear(neck_dim * 2, 1)

    def _get_loss_fn(self):
        return self._loss

    def _loss(self, X, L, Y_s):
        """Returns the average loss per example"""
        # TODO: Do not change L types every time the loss function is called!
        # For efficiency, L would be converted to {-1,0,1} before being passed
        # into the model; for consistency in the code, we leave L in {0,1,2}
        # wherever the user deals with it.

        abstains = L == 0
        # To turn off masking:
        # abstains = torch.ones_like(L).byte()

        L = L.clone()
        L[L == 0] = 0.5  # Abstains are ambivalent (0 logit)
        L[L == 2] = 0  # 2s are negative class

        neck = self.body(X)

        # Execute L head
        L_logits = self.forward_L(neck)
        # Weight loss by lf weights if applicable and mask abstains
        L_loss_masked = (
            self.L_criteria(L_logits, L.float())
            .masked_fill(abstains, 0)
            .sum(dim=1)
        )
        # Get average L loss by example per lf
        L_loss = torch.mean(L_loss_masked, dim=0) / self.m

        # Execute Y offline head
        Y_off_logits = self.forward_Y_off(neck)
        Y_off_loss = self.Y_criteria(Y_off_logits.squeeze(), Y_s[:, 0].float())

        # Execute Y online head
        Y_on_logits = self.forward_Y_on(neck, L_logits)
        Y_on_loss = self.Y_criteria(Y_on_logits.squeeze(), Y_s[:, 0].float())

        # Combine losses
        loss = (
            (self.L_head_weight * L_loss)
            + (self.Y_head_weight * Y_off_loss)
            + ((1 - self.L_head_weight - self.Y_head_weight) * Y_on_loss)
        )
        return loss

    def forward_L(self, neck):
        """Returns the logits of the L_head"""
        return self.L_head(neck)

    def forward_Y_off(self, neck):
        """Returns the logits of the offline Y_head"""
        return self.Y_head_off(neck)

    def forward_Y_on(self, neck, L_logits):
        """Returns the logits of the online Y_head"""
        L_logits = self.forward_L(neck)
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
        # Concatentate this with the output of Y_head_off
        double_neck = torch.cat((R, S), 1)
        return self.Y_head_on(double_neck)
        # neck = R + S
        # return self.Y_head_on(F.relu(torch.cat((neck, -neck), -1)))

    @torch.no_grad()
    def predict_L_proba(self, X):
        """A convenience function that predicts L probabilities"""
        return torch.sigmoid(self.L_head(self.body(X)))

    @torch.no_grad()
    def predict_Y_proba(self, X):
        """A convenience function that predicts Y offline probabilities"""
        neck = self.body(X)
        preds = torch.sigmoid(self.forward_Y_off(neck)).data.cpu().numpy()
        return np.hstack((preds, 1 - preds))

    @torch.no_grad()
    def predict_proba(self, X):
        neck = self.body(X)
        L_logits = self.forward_L(neck)
        preds = (
            torch.sigmoid(self.forward_Y_on(neck, L_logits)).data.cpu().numpy()
        )
        return np.hstack((preds, 1 - preds))


def reset_parameters(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass
