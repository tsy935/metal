import torch
import torch.nn.functional as F

from metal.end_model import IdentityModule
from metal.mmtl.modules import MetalModule, MetalModuleWrapper
from metal.mmtl.scorer import Scorer
from metal.mmtl.task import ClassificationTask
from metal.utils import convert_labels


def output_hat_func(X):
    """ Converts 1-dim output back to categorical probabilities for Metal compatibility. """
    probs = F.sigmoid(X["data"])
    return torch.cat((probs, 1 - probs), dim=1)


def categorical_cross_entropy(X, Y):
    return F.binary_cross_entropy(
        F.sigmoid(X["data"]), convert_labels(Y, "categorical", "onezero").float()
    )


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
        scorer=Scorer(standard_metrics=["accuracy"]),
        is_slice=False,
    ) -> None:

        if head_module.out_features != 1:
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

        # Add additional `is_slice` attribute
        self.is_slice = is_slice

    def __repr__(self):
        """Overrides existing __repr__ function to include slice information."""

        cls_name = type(self).__name__
        slice_repr = ", is_slice" if self.is_slice else ""
        repr = (
            f"{cls_name}(name={self.name}"
            f", loss_multiplier={self.loss_multiplier}"
            f"{slice_repr})"  # show "is_slice" if applicable
        )
        return repr
