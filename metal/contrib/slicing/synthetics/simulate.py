"""Runs simulations over equal weights, manual reweighting,
and attention based models.

Sample command:
python simulate.py --var cov --save-dir results/test --n 50 --x-range 0.6 0.7 0.8 0.9 1.0
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from time import strftime, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from metal.contrib.logging.tensorboard import TensorBoardWriter
from metal.contrib.slicing.online_dp import (
    LinearModule,
    MLPModule,
    SliceDPModel,
)
from synthetics_utils import generate_synthetic_data, plot_slice_scores

sys.path.append("/dfs/scratch0/vschen/metal")


def train_models(
    train_data,
    test_data,
    accs,
    input_module_class,
    init_kwargs,
    train_kwargs,
    verbose=False,
    use_cuda=False,
    tensorboard_logdir=None,
):
    """
    Trains baseline, oracle, and attention model
    Args:
        - train_data: (X, L) data for slice model to train on
        - test_data: (X, Y) data for validation
        - accs: [list of floats] accuracies for LFs
        - input_module_class: [nn.Module] uninitialized input module class for slice model
        - init_kwargs [dict]: kwargs to initialize input_module_class
    Returns:
        - model_[0,1,2]: trained baseline, oracle, and attention model
    """

    X, L = train_data
    m = np.shape(L)[1]  # num LFs
    d = X.shape[1]  # num features

    # baseline model, no attention
    r = 2
    init_kwargs.update({"output_dim": r})
    uniform_model = SliceDPModel(
        input_module_class(**init_kwargs),
        accs,
        r=r,
        rw=False,
        verbose=verbose,
        use_cuda=use_cuda,
    )
    curr_time = strftime("%H_%M_%S")
    log_writer = (
        TensorBoardWriter(
            log_dir=tensorboard_logdir, run_dir=f"uniform_{curr_time}"
        )
        if tensorboard_logdir
        else None
    )
    uniform_model.train_model(
        train_data, dev_data=test_data, log_writer=log_writer, **train_kwargs
    )

    # manual reweighting
    # currently hardcode weights so LF[-1] has double the weight
    weights = np.ones(m, dtype=np.float32)
    weights[-1] = 2.0
    r = 2
    init_kwargs.update({"output_dim": r})
    manual_model = SliceDPModel(
        input_module_class(**init_kwargs),
        accs,
        r=r,
        rw=False,
        L_weights=weights,
        verbose=verbose,
        use_cuda=use_cuda,
    )
    log_writer = (
        TensorBoardWriter(
            log_dir=tensorboard_logdir, run_dir=f"manual_{curr_time}"
        )
        if tensorboard_logdir
        else None
    )
    manual_model.train_model(
        train_data, dev_data=test_data, log_writer=log_writer, **train_kwargs
    )

    # our model, with attention
    r = 2
    init_kwargs.update({"output_dim": r})
    attention_model = SliceDPModel(
        input_module_class(**init_kwargs),
        accs,
        r=r,
        rw=True,
        verbose=verbose,
        use_cuda=use_cuda,
    )
    log_writer = (
        TensorBoardWriter(
            log_dir=tensorboard_logdir, run_dir=f"attention_{curr_time}"
        )
        if tensorboard_logdir
        else None
    )
    attention_model.train_model(
        train_data, dev_data=test_data, log_writer=log_writer, **train_kwargs
    )

    return uniform_model, manual_model, attention_model


def eval_model(model, data, eval_dict):
    """Evaluates models according to indexes in 'eval_dict'
    Args:
        model: trained model to evaluate
        data: (X,Y) full test set to evaluate on
        eval_dict: mapping eval slice {"slice_name":idx}
            where idx is list of indexes for corresponding slice
    Returns:
        results_dict: mapping {"slice_name": scores}
            includes "overall" accuracy by default
    """
    slice_scores = {}
    for slice_name, eval_idx in eval_dict.items():
        slice_scores[slice_name] = model.score_on_slice(
            data, eval_idx, metric="accuracy", verbose=False
        )

    slice_scores["overall"] = model.score(
        data, metric="accuracy", verbose=False
    )
    return slice_scores


def simulate(data_config, generate_data_fn, experiment_config):
    """Simulates models comparing baseline, manual, and attention models
    over the specified config.

    Args:
        config: for data generation
        generate_data_fn: data generation fn that accepts config, x_var, x_val
            for overwriting values
    Returns: (baseline_scores, manual_scores, attention_scores)
    """

    # to collect scores for all models
    baseline_scores, manual_scores, attention_scores = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )

    # get config variables
    num_trials = experiment_config["num_trials"]
    x_range = experiment_config["x_range"]
    var_name = experiment_config["x_var"]

    # for each value, run num_trials simulations
    for x in x_range:
        print(f"Simulating: {var_name}={x}")
        for _ in tqdm(range(num_trials)):

            # generate data
            X, Y, C, L = generate_synthetic_data(data_config, var_name, x)

            # convert to multiclass labels
            if -1 in Y:
                Y[Y == -1] = 2

            # train the models
            X = torch.from_numpy(X.astype(np.float32))
            L = torch.from_numpy(L.astype(np.float32))
            Y = torch.from_numpy(Y.astype(np.float32))
            split_idx = int(len(X) * experiment_config["train_prop"])
            X_train, X_test = X[:split_idx], X[split_idx:]
            Y_train, Y_test = Y[:split_idx], Y[split_idx:]
            L_train, L_test = L[:split_idx], L[split_idx:]
            C_train, C_test = C[:split_idx], C[split_idx:]

            train_data = (X_train, L_train)
            test_data = (X_test, Y_test)

            logdir = experiment_config.get("tensorboard_logdir")
            if logdir is not None:
                logdir = os.path.join(logdir, f"{var_name}_{x}")
            baseline_model, manual_model, attention_model = train_models(
                train_data,
                test_data,
                data_config["accs"],
                MLPModule,
                experiment_config["input_module_kwargs"],
                experiment_config["train_kwargs"],
                tensorboard_logdir=logdir,
            )

            # score the models
            S0_idx, S1_idx, S2_idx = (
                np.where(C_test == 0)[0],
                np.where(C_test == 1)[0],
                np.where(C_test == 2)[0],
            )
            eval_dict = {"S0": S0_idx, "S1": S1_idx, "S2": S2_idx}
            baseline_scores[x].append(
                eval_model(baseline_model, test_data, eval_dict)
            )
            manual_scores[x].append(
                eval_model(manual_model, test_data, eval_dict)
            )
            attention_scores[x].append(
                eval_model(attention_model, test_data, eval_dict)
            )

    return baseline_scores, manual_scores, attention_scores


data_config = {
    # data generation
    "N": 10000,  # num data points
    "mus": [
        np.array([-3, 0]),  # Mode 1: Y = -1
        np.array([3, 0]),  # Mode 2: Y = 1
    ],
    "labels": [-1, 1],  # labels of each slice
    "props": [0.25, 0.75],  # proportion of data in each mode
    "variances": [1, 2],  # proportion of data in each mode
    "head_config": {
        "h": 4,  # horizontal shift of slice
        "k": 0,  # vertical shift of slice
        "r": 1,  # radius of slice
        "slice_label": -1,
    },
    "accs": np.array([0.9, 0.9, 0.9]),  # default accuracy of LFs
    "covs": np.array([0.9, 0.9, 0.9]),  # default coverage of LFs
}

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    # TODO: fix warnings

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variable",
        choices=["op", "acc", "cov"],
        help="variable we are varying in simulation",
    )
    parser.add_argument("--save-dir", type=str, help="where to save results")
    parser.add_argument(
        "--n", type=int, default=25, help="num trials to run of simulation"
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs="+",
        default=None,
        help="range of values to scan over",
    )
    args = parser.parse_args()

    # define simulation config
    experiment_config = {
        "num_trials": args.n,
        "x_range": (
            np.linspace(0, 1.0, 5)
            if args.x_range is None
            else list(args.x_range)
        ),
        "x_var": args.variable,
        "input_module_kwargs": {
            "input_dim": 2,
            "middle_dims": [50, 50, 50],
            "bias": True,
        },
        "train_kwargs": {
            "batch_size": 1000,
            "n_epochs": 50,
            "print_every": 10,
            "validation_metric": "accuracy",
            "disable_prog_bar": True,
            "l2": 1e-5,
        },
        "checkpoint_runway": 5,
        "train_prop": 0.8,
        "tensorboard_logdir": args.save_dir,
    }

    # run simulations
    baseline_scores, manual_scores, attention_scores = simulate(
        data_config, generate_synthetic_data, experiment_config
    )

    # save scores and plot
    results = {
        "baseline": dict(baseline_scores),
        "manual": dict(manual_scores),
        "attention": dict(attention_scores),
    }
    print(f"Saving to {args.save_dir}")
    results_path = os.path.join(args.save_dir, f"{args.variable}-results.json")
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(results, open(results_path, "w"))
    if args.variable == "op":
        xlabel = "Overlap Proportion"
    elif args.variable == "acc":
        xlabel = "Head Accuracy"
    elif args.variable == "cov":
        xlabel = "Head Coverage"

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plot_slice_scores(results, "S2", xlabel=xlabel)
    plt.subplot(2, 2, 2)
    plot_slice_scores(results, "S1", xlabel=xlabel)
    plt.subplot(2, 2, 3)
    plot_slice_scores(results, "S0", xlabel=xlabel)
    plt.subplot(2, 2, 4)
    plot_slice_scores(results, "overall", xlabel=xlabel)
    plt.savefig(os.path.join(args.save_dir, "results.png"))
