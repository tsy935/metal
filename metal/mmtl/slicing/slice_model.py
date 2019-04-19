from collections import defaultdict

import torch
from torch.nn import functional as F

from metal.end_model import IdentityModule
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.modules import unwrap_module
from metal.utils import move_to_device, recursive_merge_dicts, set_seed


def validate_slice_tasks(tasks):
    # validate slice head cardinality
    for t in tasks:
        if t.head_module.module.out_features != 1:
            raise ValueError(
                f"{t.name}'s head_module is invalid. "
                f"SliceModel only supports binary classification "
                "specified by a output_dim=1."
            )

    base_tasks = [t for t in tasks if not t.is_slice]
    slice_tasks = [t for t in tasks if t.is_slice]

    # validate base task
    if len(base_tasks) != 1:
        raise ValueError(f"SliceModel only supports 1 base task.")
    base_task = base_tasks[0]

    # validate shared body representations
    # TODO: clean up these checks
    for t in slice_tasks:
        same_input = t.input_module is base_task.input_module
        same_middle = (
            (t.middle_module is None and base_task.middle_module is None)
            or t.middle_module is base_task.middle_module
            or (
                isinstance(unwrap_module(t.middle_module.module), IdentityModule)
                and isinstance(unwrap_module(base_task.middle_module), IdentityModule)
            )
        )
        same_attention = (
            (t.attention_module is None and base_task.attention_module is None)
            or t.attention_module is base_task.attention_module
            or (
                isinstance(unwrap_module(t.attention_module.module), IdentityModule)
                and isinstance(
                    unwrap_module(base_task.attention_module), IdentityModule
                )
            )
        )

        has_same_body = same_input and same_middle and same_attention

        if not has_same_body:
            raise ValueError(
                f"Slice tasks must have the same body as base task: "
                f"(same_input={same_input}"
                f", same_middle={same_middle}"
                f", same_attention={same_attention})"
            )

    # validate that one of the "slice_tasks" operates on the entire dataset
    if not any([t.name.endswith(":BASE") for t in slice_tasks]):
        raise ValueError(
            "There must be a `slice_task` designated to operate "
            f"on the entire labelset with name '{base_task.name}:BASE'."
        )


class SliceModel(MetalModel):
    """ Slice-aware version of MetalModel.

    At the moment, only supports:
        * Binary classification heads (with output_dim=1, breaking Metal convention)
        * A single base task + an arbitrary number of slice tasks (is_slice=True)
    """

    def __init__(self, tasks, **kwargs):
        validate_slice_tasks(tasks)

        super().__init__(tasks, **kwargs)
        self.base_task = [t for t in self.task_map.values() if not t.is_slice][0]
        self.slice_tasks = {name: t for name, t in self.task_map.items() if t.is_slice}

        # At the moment, we assign the slice weights = 1/num_slices
        # The base_task maintains its default multiplier = 1.0
        # TODO: look into more rigorous way to do this
        slice_weight = 1.0 / len(self.slice_tasks) if self.slice_tasks else 1.0
        for t in self.slice_tasks.values():
            t.loss_multiplier = slice_weight

    def forward_body(self, X):
        """ Makes a forward pass through the "body" of the network
        (everything before the head)."""

        input = move_to_device(X, self.config["device"])
        base_task_name = self.base_task.name

        # Extra .module because of DataParallel wrapper!
        input_module = self.input_modules[base_task_name].module
        middle_module = self.middle_modules[base_task_name].module
        attention_module = self.attention_modules[base_task_name].module

        out = attention_module(middle_module(input_module(input)))
        return out

    def forward_heads(self, body, task_names):
        """ Given body (everything before head) representation, return dict of
        task_head outputs for specified task_names """

        return {t: self.head_modules[t].module(body) for t in task_names}

    def forward_attention_logits(self, slice_heads, slice_task_names):
        """ Computes unnomralized slice_head attention weights """

        # slice_preds is the [batch_size, num_slices] concatenated prediction for
        # each slice head
        slice_preds = torch.cat(
            [slice_heads[slice_name]["data"] for slice_name in slice_task_names], dim=1
        )

        # A is the [batch_size, num_slices] unormalized Tensor representing the
        # amount of attention to pay to each head (based on each one's confidence)
        A_weights = abs(slice_preds)
        return A_weights

    def forward(self, X, task_names):
        """ Perform forward pass with slice-reweighted base representation through
        the base_task head. """

        # Sort names to ensure that the representation is always
        # computed in the same order
        slice_task_names = sorted(
            [slice_task_name for slice_task_name in self.slice_tasks.keys()]
        )

        body = self.forward_body(X)
        slice_heads = self.forward_heads(body, slice_task_names)

        # body_rep is the representation that would normally go into the task_heads
        body_rep = body["data"]

        # [batch_size, num_slices] unnormalized attention weights.
        # we then normalize the weights across all heads
        A_weights = self.forward_attention_logits(slice_heads, slice_task_names)
        A = F.softmax(A_weights, dim=1)

        # slice_weights is the [num_slices, body_dim] concatenated tensor
        # representing linear transforms for the body into the head values
        slice_weights = torch.cat(
            [
                # TODO: fix the need for a double module (DataParllel + MetalModule)...
                self.head_modules[t].module.module.weight
                for t in slice_task_names
            ],
            dim=0,
        )

        # we reweight the linear mappings `slice_weights` by the
        # attention scores `A` and use this to reweight the base representation
        reweighted_rep = (A @ slice_weights) * body_rep

        # finally, forward through original task head with reweighted representation
        body["data"] = reweighted_rep

        # compute losses for individual slices + reweighted Y_head
        output = self.forward_heads(body, [self.base_task.name])
        output.update(slice_heads)
        return output

    @torch.no_grad()
    def calculate_attention(self, X):
        """ Retrieve the unnormalized attention weights used in the model over
        slice_heads

        Borrows template from MetalModel.calculate_probs """

        assert self.eval()
        slice_task_names = sorted(
            [slice_task_name for slice_task_name in self.slice_tasks.keys()]
        )

        body = self.forward_body(X)
        slice_heads = self.forward_heads(body, slice_task_names)
        A_weights = self.forward_attention_logits(slice_heads, slice_task_names)
        return A_weights

    def attention_with_gold(self, payload):
        """ Retrieve the attention weights used in the model for viz/debugging

        Borrows template from MetalModel.predict_with_gold """

        Ys = defaultdict(list)
        A_weights = []

        for batch_num, (Xb, Yb) in enumerate(payload.data_loader):
            Ab = self.calculate_attention(Xb)
            A_weights.extend(Ab.cpu().numpy())

            # NOTE: this is redundant, because we only have one set of A_weights
            # but we keep this format to match the expected format of Ys
            for label_name, yb in Yb.items():
                Ys[label_name].extend(yb.cpu().numpy())

        return Ys, A_weights
