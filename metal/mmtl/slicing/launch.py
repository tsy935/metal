"""
Creates slice/baseline/ablation model, trains, and evaluates on
corresponding slice prediction labelsets.

python launch.py --seed 1 --tasks RTE --slice_dict '{"RTE": ["dash_semicolon", "more_people", "BASE"]}' --model_type naive --n_epochs 1

python launch.py --model_type manual --tasks COLA --lr 5e-05 --lr_scheduler linear --checkpoint_metric COLA/COLA_valid/COLA_gold/matthews_corr --slice_dict '{"COLA": ["short_premise", "has_wh_words"]}' --slice_loss_mult '{"COLA_slice:short_premise:pred": 5}'

"""

import argparse
import copy
import json
import os
from pprint import pprint

import numpy as np
import torch

from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads, task_defaults
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.slicing.moe_model import MoEModel
from metal.mmtl.slicing.slice_model import SliceModel, SliceQPModel, SliceRepModel
from metal.mmtl.slicing.tasks import convert_to_slicing_tasks
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults
from metal.utils import add_flags_from_config, recursive_merge_dicts

# Overwrite defaults
task_defaults["attention"] = False
model_defaults["verbose"] = False
model_defaults["delete_heads"] = True  # mainly load the base representation weights

# by default, log last epoch (not best)
trainer_defaults["checkpoint"] = True
trainer_defaults["checkpoint_config"]["checkpoint_best"] = False
trainer_defaults["writer"] = "tensorboard"

# Model configs
model_configs = {
    "naive": {"model_class": MetalModel, "active_slice_heads": {}},
    "hard_param": {
        "model_class": MetalModel,
        "active_slice_heads": {"pred": True, "ind": False},
    },
    "manual": {
        "model_class": MetalModel,
        "active_slice_heads": {"pred": True, "ind": False},
    },
    "soft_param": {
        "model_class": SliceModel,
        "active_slice_heads": {"pred": True, "ind": True},
    },
    "soft_param_rep": {
        "model_class": SliceRepModel,
        "active_slice_heads": {"pred": False, "ind": True},
    },
    "slice_qp_model": {
        "model_class": SliceQPModel,
        "active_slice_heads": {"pred": False, "shared_pred": True, "ind": True},
    },
    "moe": {
        "model_class": MoEModel,
        "active_slice_heads": {"pred": True, "ind": False},
    },
}


def main(args):
    # Extract flags into their respective config files
    trainer_config = recursive_merge_dicts(
        trainer_defaults, vars(args), misses="ignore"
    )
    model_config = recursive_merge_dicts(model_defaults, vars(args), misses="ignore")
    task_config = recursive_merge_dicts(task_defaults, vars(args), misses="ignore")

    task_names = args.tasks.split(",")
    assert len(task_names) == 1
    base_task_name = task_names[0]

    # Default name for log directory to task names
    if args.run_name is None:
        run_name = f"{args.model_type}_{args.tasks}"
        trainer_config["writer_config"]["run_name"] = run_name

    # Get model configs
    config = model_configs[args.model_type]
    active_slice_heads = config["active_slice_heads"]
    model_class = config["model_class"]

    slice_dict = json.loads(args.slice_dict) if args.slice_dict else {}
    task_config.update({"slice_dict": slice_dict})
    task_config["active_slice_heads"] = active_slice_heads

    # Create tasks and payloads
    tasks, payloads = create_glue_tasks_payloads(task_names, **task_config)

    # Create evaluation payload with test_slices -> primary task head
    task_config.update({"slice_dict": slice_dict})
    task_config["active_slice_heads"] = {
        # turn pred labelsets on, and use model's value for ind head
        "pred": True,
        "ind": active_slice_heads.get("ind", False),
    }

    # Initialize trainer
    slice_tasks, slice_payloads = create_glue_tasks_payloads(task_names, **task_config)
    pred_labelsets = [
        labelset
        for labelset in slice_payloads[0].labels_to_tasks.keys()
        if "pred" in labelset or "_gold" in labelset
    ]
    # Only eval "pred" labelsets on main task head -- continue eval of inds on ind-heads
    for p in slice_payloads[1:]:  # remap val and test payloads
        p.remap_labelsets(
            {pred_labelset: base_task_name for pred_labelset in pred_labelsets}
        )

    if args.validate_on_slices:
        print("Will compute validation scores for slices based on main head.")
        payloads[1] = slice_payloads[1]

    if active_slice_heads:
        tasks = convert_to_slicing_tasks(tasks)

    if args.model_type == "manual":
        slice_loss_mult = (
            json.loads(args.slice_loss_mult) if args.slice_loss_mult else {}
        )
        for task in tasks:
            if task.name in slice_loss_mult.keys():
                task.loss_multiplier *= slice_loss_mult[task.name]
                print(
                    "Override {} loss multiplier with{}.".format(
                        task.name, slice_loss_mult[task.name]
                    )
                )

    # Initialize and train model
    if args.model_type == "moe":
        experts = {}
        for model_num, slice_name in enumerate(slice_dict[base_task_name]):
            task_config.update({"slice_dict": {base_task_name: [slice_name]}})
            tasks_slice, payloads_slice = create_glue_tasks_payloads(
                task_names, **task_config
            )
            tasks_slice = convert_to_slicing_tasks(tasks_slice)
            print(tasks_slice)
            print(payloads_slice)

            # remove the base task labels from the expert payloads.
            for p in payloads_slice:
                p.labels_to_tasks.pop(f"{base_task_name}_gold")
            # remove the slice task labels from the payloads used to train the MoEModel.
            for p in payloads:
                p.labels_to_tasks.pop(f"{base_task_name}_slice:{slice_name}:pred")

            # rotate through all GPUs to allocate with models!
            device = model_num % torch.cuda.device_count()
            # remove the first task (main task)
            model = MetalModel(tasks_slice[1:], verbose=False, device=device)
            trainer = MultitaskTrainer(seed=args.seed)
            metrics_dict = trainer.train_model(model, payloads_slice, **trainer_config)
            print(metrics_dict)
            experts[slice_name] = model
        # MoEModel takes one base task
        model = model_class([tasks[0]], experts, **model_config)
    else:
        # Initialize and train model
        model = model_class(tasks, **model_config)

    # train model
    trainer = MultitaskTrainer(**trainer_config)
    trainer.train_model(model, payloads)

    # write configs
    trainer._set_writer()
    trainer.writer.write_config(model_config, "model_config")
    trainer.writer.write_config(task_config, "task_config")
    trainer.writer.write_config(vars(args), "args")

    # Evaluate trained model on slices
    model.eval()
    slice_metrics = model.score(slice_payloads[2])
    pprint(slice_metrics)
    if trainer.writer:
        trainer.writer.write_metrics(slice_metrics, "slice_metrics.json")
        return os.path.join(trainer.writer.log_subdir, "slice_metrics.json")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Launch slicing models/baselines on GLUE tasks", add_help=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=np.random.randint(1e6),
        help="A single seed to use for trainer, model, and task configs",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(model_configs.keys()),
        help="Model to run and evaluate",
    )
    parser.add_argument(
        "--validate_on_slices",
        type=bool,
        default=False,
        help="Whether to map eval main head on validation set during training",
    )

    parser.add_argument(
        "--slice_loss_mult",
        type=str,
        default=None,
        help="Slice loss multipliers that override the default ones (1/num_slices).",
    )
    parser = add_flags_from_config(parser, trainer_defaults)
    parser = add_flags_from_config(parser, model_defaults)
    parser = add_flags_from_config(parser, task_defaults)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
