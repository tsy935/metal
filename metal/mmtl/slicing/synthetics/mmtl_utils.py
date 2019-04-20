import torch
import torch.nn as nn

from metal.mmtl.data import MmtlDataLoader, MmtlDataset
from metal.mmtl.payload import Payload
from metal.mmtl.slicing.synthetics.data_generator import (
    generate_data,
    generate_slice_labels,
)
from metal.mmtl.slicing.tasks import BinaryClassificationTask, create_slice_task


def create_tasks(task_name, slice_names=[]):
    input_module = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
    head_module = nn.Linear(5, 1)  # NOTE: slice_model requires 1dim output head
    task = BinaryClassificationTask(
        name=task_name, input_module=input_module, head_module=head_module
    )
    tasks = [task]

    for slice_name in slice_names:
        slice_task_name = f"{task_name}:{slice_name}"
        slice_task = create_slice_task(task, slice_task_name)
        tasks.append(slice_task)

    return tasks


def create_payloads(
    task_name,
    uid_lists,
    Xs,
    Ys,
    batch_size=1,
    slice_funcs={},
    SPLITS=["train", "valid", "test"],
):

    payloads = []
    labels_to_tasks = {"labelset_gold": task_name}
    for i, split in enumerate(SPLITS):
        payload_name = f"payload_{split}"

        # convert to torch tensors
        X_dict = {"data": torch.Tensor(Xs[i]), "uids": torch.Tensor(uid_lists[i])}
        Y_dict = {"labelset_gold": torch.Tensor(Ys[i])}

        if slice_funcs:
            slice_labels = generate_slice_labels(Xs[i], Ys[i], slice_funcs)

            for slice_name, slice_label in slice_labels.items():
                slice_task_name = f"{task_name}:{slice_name}"
                slice_labelset_name = f"labelset:{slice_name}"
                Y_dict[slice_labelset_name] = torch.tensor(slice_label)
                labels_to_tasks[slice_labelset_name] = slice_task_name

        dataset = MmtlDataset(X_dict, Y_dict)
        data_loader = MmtlDataLoader(dataset, batch_size=batch_size)
        payload = Payload(payload_name, data_loader, labels_to_tasks, split)
        payloads.append(payload)

    return payloads
