import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.end_model.em_defaults import em_default_config
from metal.end_model.end_model import EndModel
from metal.metrics import metric_score
from metal.utils import recursive_merge_dicts
from metal.end_model.loss import SoftCrossEntropyLoss


class LinearModule(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.input_layer(x)


class MLPModule(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dims=[], bias=False):
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

        # No bias-- only learn weights for L_head
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
        self.Y_head = nn.Linear(y_d, self.output_dim, bias=False)

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
        L[L!=0] = 1         
        
        # LF heads loss
        loss_1 = torch.mean(
            self.criteria_L(self.forward_L(X), L) @ self.L_weights
        )

        # Mask if none of the slices are activated-- don't backprop
        dp_head_mask = (
            (torch.sum(abs(L), dim=1) > 0).float()
        )
        masked_Y_loss = self.criteria_Y(self.forward_Y(X), Y_weak) * dp_head_mask
        loss_2 = torch.mean(masked_Y_loss)

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

