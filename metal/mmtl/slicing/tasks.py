import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.end_model import IdentityModule
from metal.mmtl.glue.glue_metrics import acc_f1
from metal.mmtl.modules import MetalModule, MetalModuleWrapper, unwrap_module
from metal.mmtl.scorer import Scorer
from metal.mmtl.task import ClassificationTask, RegressionTask
from metal.utils import convert_labels


def create_slice_task(base_task, slice_task_name, slice_head_type, loss_multiplier=1.0):
    """Creates a slice task identical to a base task but with different head params"""
    # for pred head, copy the base task head to match the output labelspace
    if slice_head_type == "pred":
        slice_task = copy.copy(base_task)
        slice_task.name = slice_task_name
        slice_task.head_module = copy.deepcopy(base_task.head_module)
        slice_task.slice_head_type = slice_head_type
        slice_task.loss_multiplier = loss_multiplier

    # for ind heads, always initialize a binary class predictor
    elif slice_head_type == "ind":
        slice_task = BinaryClassificationTask(
            slice_task_name,
            input_module=unwrap_module(base_task.input_module),
            middle_module=unwrap_module(base_task.middle_module),
            attention_module=unwrap_module(base_task.attention_module),
            head_module=unwrap_module(base_task.head_module),
            loss_multiplier=loss_multiplier,
            slice_head_type=slice_head_type,
        )
    else:
        raise ValueError(f"Invalid slice_head_type: {slice_head_type}")

    return slice_task


def convert_to_slicing_tasks(tasks):
    """ Converts list of existing tasks into Slice-ready tasks (if applicable). """
    slicing_tasks = []
    for t in tasks:

        input_module = unwrap_module(t.input_module)
        middle_module = unwrap_module(t.middle_module)
        attention_module = unwrap_module(t.attention_module)
        head_module = unwrap_module(t.head_module)

        # change all output_dims -> 1
        if isinstance(head_module, torch.nn.Linear):
            if head_module.out_features != 1:
                print(
                    f"Modifying {t.name} out_features from {head_module.out_features} -> 1"
                )
                head_module = nn.Linear(head_module.in_features, 1)
        else:
            raise ValueError(f"head_module for {t.name} is not a valid FC layer.")

        if not hasattr(t, "slice_head_type"):
            head_type = None
        else:
            head_type = t.slice_head_type

        if isinstance(t, ClassificationTask):

            slice_t = BinaryClassificationTask(
                t.name,
                input_module,
                middle_module,
                attention_module,
                head_module,
                output_hat_func=t.output_hat_func,
                loss_hat_func=t.loss_hat_func,
                loss_multiplier=t.loss_multiplier,
                scorer=t.scorer,
                slice_head_type=head_type,
            )
            slicing_tasks.append(slice_t)

        elif isinstance(t, RegressionTask):
            slice_t = SliceRegressionTask(
                t.name,
                input_module,
                middle_module,
                attention_module,
                head_module,
                output_hat_func=t.output_hat_func,
                loss_hat_func=t.loss_hat_func,
                loss_multiplier=t.loss_multiplier,
                scorer=t.scorer,
                slice_head_type=head_type,
            )
            slicing_tasks.append(slice_t)

        else:
            raise ValueError(f"{t.__class__.__name__} is not supported!")

    return slicing_tasks


def output_hat_func(X):
    """ Converts 1-dim output back to categorical probabilities for Metal compatibility. """
    probs = torch.sigmoid(X["data"])
    return torch.cat((probs, 1 - probs), dim=1)


def categorical_cross_entropy(X, Y):
    return F.binary_cross_entropy(
        torch.sigmoid(X["data"]), convert_labels(Y, "categorical", "onezero").float()
    )


class SliceRegressionTask(RegressionTask):
    """A regression task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        attention_module=IdentityModule(),
        head_module=IdentityModule(),
        output_hat_func=(lambda X: X["data"]),
        # Note: no sigmoid (target labels can be in any range)
        loss_hat_func=(lambda X, Y: F.mse_loss(X["data"].view(-1), Y.view(-1))),
        loss_multiplier=1.0,
        scorer=Scorer(standard_metrics=[]),
        slice_head_type=None,
    ) -> None:

        if (
            head_module
            and not isinstance(head_module, IdentityModule)
            and head_module.out_features != 1
        ):
            raise ValueError(f"{self.__class__.__name__} must have an output dim 1.")
        super(SliceRegressionTask, self).__init__(
            name,
            input_module,
            middle_module,
            attention_module,
            head_module,
            output_hat_func,
            loss_hat_func,
            loss_multiplier,
            scorer,
        )

        # Add an additional attribute to indicator head type
        assert slice_head_type in ["ind", "pred", None]
        self.slice_head_type = slice_head_type


class BinaryClassificationTask(ClassificationTask):
    """A binary classification task supported in a SliceModel.

    Key changes:
    * Adds an additional field, `is_slice`, which indicates whether the
        task targets dataset slice.
    * Enforces that the output dimension of the head=1. This is necessary
        for the current design of attentino in the SliceModel.
    """

    def __init__(
        self,
        name,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        attention_module=IdentityModule(),
        head_module=IdentityModule(),
        output_hat_func=output_hat_func,
        loss_hat_func=categorical_cross_entropy,
        loss_multiplier=1.0,
        scorer=Scorer(custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}),
        # standard_metrics=["accuracy"]),
        slice_head_type=None,
    ) -> None:

        if (
            head_module
            and not isinstance(head_module, IdentityModule)
            and head_module.out_features != 1
        ):
            raise ValueError(f"{self.__class__.__name__} must have an output dim 1.")

        super(BinaryClassificationTask, self).__init__(
            name,
            input_module,
            middle_module,
            attention_module,
            head_module,
            output_hat_func,
            loss_hat_func,
            loss_multiplier,
            scorer,
        )

        # Add an additional attribute to indicator head type
        assert slice_head_type in ["ind", "pred", None]
        self.slice_head_type = slice_head_type

    def __repr__(self):
        """Overrides existing __repr__ function to include slice information."""
        metal_repr = str(super(BinaryClassificationTask, self).__repr__())
        return (
            f"{metal_repr[:-1]}, slice_head_type={self.slice_head_type})"
        )  # trim closing paren in repr')'
