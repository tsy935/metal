import copy

import torch
import torch.nn as nn

from metal.mmtl.data import MmtlDataLoader, MmtlDataset
from metal.mmtl.modules import MetalModuleWrapper
from metal.mmtl.payload import Payload
from metal.mmtl.slicing.synthetics.data_generator import (
    generate_data,
    generate_slice_labels,
)
from metal.mmtl.slicing.tasks import BinaryClassificationTask, create_slice_task
from metal.mmtl.trainer import MultitaskTrainer


def create_tasks(
    task_name,
    rep_dim=5,
    slice_names=[],
    slice_weights={},
    create_ind=True,
    create_preds=True,
    create_base=True,
    create_shared_slice_pred=False,
    use_ind_module=False,
    custom_neck_dim=None,
    verbose=False,
    h_dim=None,
):
    input_module = nn.Sequential(nn.Linear(2, rep_dim), nn.ReLU())
    # NOTE: slice_model requires 1dim output head
    head_input_dim = h_dim if h_dim else rep_dim
    head_module = nn.Linear(head_input_dim, 1)

    base_task = BinaryClassificationTask(
        name=task_name, input_module=input_module, head_module=head_module
    )

    if create_shared_slice_pred:
        shared_slice_pred_task = copy.copy(base_task)
        shared_slice_head = copy.deepcopy(head_module)
        shared_slice_pred_task.head_module = MetalModuleWrapper(shared_slice_head)
        shared_slice_pred_task.slice_head_type = "pred"

    tasks = []
    # for each slice create an 'ind' task: predicts whether we are in the slice
    # and a 'pred' task: "expert" on predicting labels of slice
    for slice_name in slice_names:
        # TODO: make loss_multiplier a parameter
        # loss_multiplier = 1.0 / (2 * len(slice_names))
        if slice_name in slice_weights:
            loss_multiplier = slice_weights[slice_name]
        else:
            loss_multiplier = 1.0

        if create_shared_slice_pred:
            curr_pred_task = copy.copy(shared_slice_pred_task)
            curr_pred_task.name = f"{task_name}:{slice_name}:pred"
            tasks.append(curr_pred_task)
        elif create_preds:
            slice_pred_task = create_slice_task(
                base_task,
                f"{task_name}:{slice_name}:pred",
                "pred",
                loss_multiplier=loss_multiplier,
            )
            tasks.append(slice_pred_task)

        if create_ind:
            slice_ind_task = create_slice_task(
                base_task,
                f"{task_name}:{slice_name}:ind",
                "ind",
                loss_multiplier=loss_multiplier,
            )

            if use_ind_module:
                ind_head_module = MetalModuleWrapper(nn.Linear(head_input_dim, 1))
            else:
                ind_head_module = MetalModuleWrapper(nn.Linear(rep_dim, 1))
            slice_ind_task.head_module = ind_head_module
            tasks.append(slice_ind_task)

    if custom_neck_dim:
        # re-init task with different base head module :(
        head_module = MetalModuleWrapper(nn.Linear(custom_neck_dim, 1))
        new_base_task = copy.copy(base_task)
        new_base_task.head_module = head_module
        base_task = new_base_task

    if create_base:
        tasks.append(base_task)

    if verbose:
        print(f"Creating {len(tasks)} tasks...")
        for t in tasks:
            print(t)
    return tasks


def create_payloads(
    task_name,
    uid_lists,
    Xs,
    Ys,
    batch_size=1,
    slice_funcs={},
    SPLITS=["train", "valid", "test"],
    create_ind=True,
    create_preds=True,
    create_base=True,
    create_shared_slice_pred=False,
    verbose=False,
):

    payloads = []
    labels_to_tasks = {"labelset_gold": task_name} if create_base else {}
    for i, split in enumerate(SPLITS):
        payload_name = f"payload_{split}"

        # convert to torch tensors
        X_dict = {"data": torch.Tensor(Xs[i]), "uids": torch.Tensor(uid_lists[i])}
        Y_dict = {"labelset_gold": torch.Tensor(Ys[i])} if create_base else {}
        if create_shared_slice_pred:
            create_preds = True

        if slice_funcs:
            slice_labels = generate_slice_labels(
                Xs[i], Ys[i], slice_funcs, create_ind, create_preds
            )

            # labelset_name -> {"ind": [1,2,1,2,2], "pred": [0,1,0,2,2]}
            for slice_name, slice_label in slice_labels.items():
                # slice_type \in {"ind", "pred"}
                for slice_type, label in slice_label.items():
                    slice_task_name = f"{task_name}:{slice_name}:{slice_type}"
                    slice_labelset_name = f"labelset:{slice_name}:{slice_type}"
                    Y_dict[slice_labelset_name] = torch.tensor(label)
                    labels_to_tasks[slice_labelset_name] = slice_task_name

        dataset = MmtlDataset(X_dict, Y_dict)
        data_loader = MmtlDataLoader(dataset, batch_size=batch_size)
        payload = Payload(payload_name, data_loader, labels_to_tasks, split)
        payloads.append(payload)

    if verbose:
        print(f"Creating {len(payloads)} payloads...")
        for p in payloads:
            print(p)
    return payloads


def train_slice_experts(
    uid_lists, Xs, Ys, model_class, slice_funcs, batch_size=16, **trainer_kwargs
):
    """ For each slice_func,
        1) initializes a model using model_class(tasks),
        2) trains on payloads where data is masked by slice.
    """

    experts = {}
    task_name = "expert"
    for slice_name, slice_fn in slice_funcs.items():
        tasks = create_tasks(
            task_name,
            slice_names=[slice_name],
            create_ind=False,
            create_base=False,
            verbose=True,
        )
        payloads = create_payloads(
            task_name,
            uid_lists,
            Xs,
            Ys,
            batch_size=batch_size,
            slice_funcs={slice_name: slice_fn},
            create_ind=False,
            create_base=False,
            verbose=True,
        )
        print(tasks)
        print(payloads)
        model = model_class(tasks)
        seed = trainer_kwargs.get("seed", None)
        trainer = MultitaskTrainer(seed=seed)
        metrics_dict = trainer.train_model(model, payloads, **trainer_kwargs)
        print(metrics_dict)
        experts[slice_name] = model

    return experts
