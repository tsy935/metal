import torch
from torch.nn import functional as F

from metal.mmtl.metal_model import MetalModel
from metal.utils import move_to_device, recursive_merge_dicts, set_seed


def validate_slice_tasks(tasks):
    for t in tasks:
        if t.head_module.module.out_features != 1:
            raise ValueError(
                f"{t.name}'s head_module is invalid. "
                f"SliceModel only supports binary classification "
                "specified by a output_dim=1."
            )

    base_tasks = [t for t in tasks if not t.is_slice]
    slice_tasks = [t for t in tasks if t.is_slice]

    if len(base_tasks) != 1:
        raise ValueError(f"SliceModel only supports 1 base task.")
    base_task = base_tasks[0]

    for t in slice_tasks:
        has_same_body = (
            t.input_module is base_task.input_module
            and (
                t.middle_module is None
                and base_task.middle_module is None
                or t.middle_module is base_task.middle_module
            )
            and (
                t.attention_module is None
                and base_task.attention_module is None
                or t.attention_module is base_task.attention_module
            )
        )
        if not has_same_body:
            raise ValueError("Slice tasks must have the same body as base task.")


class SliceModel(MetalModel):
    def __init__(self, tasks, **kwargs):
        validate_slice_tasks(tasks)

        super().__init__(tasks, **kwargs)
        self.base_task = [t for t in self.task_map.values() if not t.is_slice][0]
        self.slice_tasks = {name: t for name, t in self.task_map.items() if t.is_slice}

        # TODO: more rigorously weight the slice tasks
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
        return {t: self.head_modules[t].module(body) for t in task_names}

    def forward(self, X, task_names):
        slice_task_names = [
            slice_task_name for slice_task_name in self.slice_tasks.keys()
        ]

        body = self.forward_body(X)
        slice_heads = self.forward_heads(body, slice_task_names)

        # body_rep is the representation that would normally go into the task_heads
        body_rep = body["data"]

        # slice_preds is the [batch_size, num_slices] concatenated prediction for
        # each slice head
        slice_preds = torch.cat(
            [slice_head["data"] for slice_head in slice_heads.values()], dim=1
        )

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

        # A is the [batch_size, num_slices] Tensor representing the amount of
        # attention to pay to each head (based on each one's confidence)
        A = F.softmax(abs(slice_preds), dim=1)

        # we reweight the linear mappings `slice_weights` by the
        # attention scores `A` and use this to reweight the base representation
        reweighted_rep = (A @ slice_weights) * body_rep

        # finally, forward through original task head with reweighted representation
        body["data"] = reweighted_rep

        # compute losses for individual slices + reweighted Y_head
        output = self.forward_heads(body, [self.base_task.name])
        output.update(slice_heads)
        return output
