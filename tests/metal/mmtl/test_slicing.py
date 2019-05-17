import unittest
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from metal.mmtl.slicing.slice_model import SliceRepModel
from metal.mmtl.slicing.synthetics.data_generator import get_circle_mask
from metal.mmtl.slicing.synthetics.mmtl_utils import (
    create_payloads as create_synthetics_payloads,
    create_tasks,
)
from metal.mmtl.trainer import MultitaskTrainer
from metal.utils import convert_labels, split_data

SPLITS = ["train", "valid", "test"]


def create_payloads(task_name, N, slice_funcs, batch_size=1):
    X = np.random.random((N, 2)) * 2 - 1
    Y = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    # index_fn to label for abberations in decision boundary
    label_flips = {
        partial(get_circle_mask, center=(0.25, 0), radius=0.3): 1,
        partial(get_circle_mask, center=(-0.3, -0.5), radius=0.3): 2,
    }
    for mask_fn, label in label_flips.items():
        Y[mask_fn(X)] = label

    uids = list(range(N))
    uid_lists, Xs, Ys = split_data(uids, X, Y, splits=[0.8, 0.05, 0.15], shuffle=True)

    # create mmtl-style payloads
    payloads = create_synthetics_payloads(
        task_name,
        uid_lists,
        Xs,
        Ys,
        batch_size=1000,
        slice_funcs=slice_funcs,
        verbose=False,
    )

    # evaluating on the base_task head only
    labelsets = list(payloads[0].labels_to_tasks.keys())
    pred_labelsets = [labelset for labelset in labelsets if ":pred" in labelset]
    pred_labelsets.append("labelset_gold")
    for eval_payload in payloads[1:]:  # loop through val and test
        eval_payload.remap_labelsets(
            labels_to_tasks={labelset: task_name for labelset in pred_labelsets},
            default_none=True,
            verbose=False,
        )
    return payloads


class SlicingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.trainer = MultitaskTrainer(verbose=False, lr=0.1, n_epochs=15)

    def test_slice_rep_model_synthetics(self):
        """One task with one train payload and one labelset"""
        N = 3000
        identity_fn = lambda x: np.ones(x.shape[0], dtype=np.bool)
        slice_funcs = {
            "slice_1": partial(get_circle_mask, center=(0.35, -0.2), radius=0.3),
            "slice_2": partial(get_circle_mask, center=(-0.35, -0.25), radius=0.3),
            "BASE": identity_fn,
        }

        task_name = "base_task"
        tasks = create_tasks(
            task_name,
            rep_dim=7,
            slice_names=list(slice_funcs.keys()),
            create_ind=True,
            create_preds=False,
            verbose=False,
        )
        model = SliceRepModel(tasks, verbose=False)
        payloads = create_payloads(
            task_name, N, slice_funcs=slice_funcs, batch_size=512
        )

        # make sure overall/slice-level metrics exist
        labelset_names = ["labelset_gold"]
        labelset_names += [
            f"labelset:{slice_name}:pred" for slice_name in slice_funcs.keys()
        ]

        metrics_to_check = []
        for split in SPLITS:
            for labelset_name in labelset_names:
                metrics_to_check.append(
                    f"{task_name}/payload_{split}/{labelset_name}/accuracy"
                )

        metrics_dict = self.trainer.train_model(model, payloads)

        # make sure all relevant metrics exist
        relevant_reported_metrics = set(metrics_to_check).intersection(
            set(metrics_dict.keys())
        )
        self.assertEqual(len(relevant_reported_metrics), len(metrics_to_check))

        # make sure we can fit this simple dataset
        for metric in metrics_to_check:
            score = metrics_dict[metric]
            print(metric, score)
            self.assertGreater(score, 0.9)


if __name__ == "__main__":
    unittest.main()
