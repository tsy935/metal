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
import torch
import pickle
import numpy as np

import sys, os
os.environ['METALHOME'] = '/dfs/scratch1/saelig/slicing/metal/'
sys.path.append('../../../../metal')

from metal.mmtl.birds.bird_tasks import create_birds_tasks_payloads, task_defaults
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.slicing.slice_model import *
from metal.mmtl.slicing.moe_model import MoEModel
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
        "active_slice_heads": {"pred": True, "shared_pred": False, "ind": False},
    },
    "manual": {
        "model_class": MetalModel,
        "active_slice_heads": {"pred": True, "shared_pred": False, "ind": False},
    },
    "soft_param": {
        "model_class": SliceModel,
        "active_slice_heads": {"pred": True, "shared_pred": False, "ind": True},
    },
    "soft_param_rep": {
        "model_class": SliceRepModel,
        "active_slice_heads": {"pred": False, "shared_pred": False, "ind": True},
    },
    "slice_qp_model": {
        "model_class": SliceQPModel,
        "active_slice_heads": {"pred": False, "shared_pred": True, "ind": True},
    },
    "moe": {
        "model_class": MoEModel,
        "active_slice_heads": {"pred": True, "shared_pred": False, "ind": False},
    },
}


opj = os.path.join
HOME_DIR = '/dfs/scratch1/saelig/slicing/'
DATASET_DIR = opj(HOME_DIR,'CUB_200_2011')
IMAGES_DIR = opj(DATASET_DIR, 'images')
TENSORS_DIR = opj(HOME_DIR, 'birds_data')


def get_slices():
    arr = np.loadtxt('slices.csv', delimiter=',', skiprows=1)
    slices = arr[(arr[:,2] <= 10.0) & (arr[:,2] >= 1.0)][:,0] #get slices only between 1% and 10% in size
    return slices.astype(int).tolist()

def main(args):
    print('Loading data...')
    train_image_ids = torch.load(opj(TENSORS_DIR,'train_image_ids.pt'))
    valid_image_ids = torch.load(opj(TENSORS_DIR,'valid_image_ids.pt'))
    test_image_ids = torch.load(opj(TENSORS_DIR,'test_image_ids.pt'))
    X_train = torch.load(opj(TENSORS_DIR,'X_train.pt'))
    X_valid = torch.load(opj(TENSORS_DIR,'X_valid.pt'))
    X_test = torch.load(opj(TENSORS_DIR,'X_test.pt'))
    Y_train = torch.load(opj(TENSORS_DIR,'Y_train.pt'))
    Y_valid = torch.load(opj(TENSORS_DIR,'Y_valid.pt'))
    Y_test = torch.load(opj(TENSORS_DIR,'Y_test.pt'))
    attrs_dict = pickle.load(open(opj(TENSORS_DIR, 'attrs_dict.pkl'),'rb'))
    print('Done')

    image_id_splits = train_image_ids, valid_image_ids, test_image_ids
    X_splits = X_train, X_valid, X_test
    Y_splits = Y_train, Y_valid, Y_test

    # Extract flags into their respective config files    
    trainer_config = recursive_merge_dicts(
        trainer_defaults, vars(args), misses="ignore"
    )
    model_config = recursive_merge_dicts(model_defaults, vars(args), misses="ignore")
    task_config = recursive_merge_dicts(task_defaults, vars(args), misses="ignore")

    # task_names = args.tasks.split(",")
    # assert len(task_names) == 1
    base_task_name = 'BirdClassificationTask'

    # Default name for log directory to task names
    if args.run_name is None:
        run_name = f"{args.model_type}_{base_task_name}"
        trainer_config["writer_config"]["run_name"] = run_name

    # Get model configs
    config = model_configs[args.model_type]
    active_slice_heads = config["active_slice_heads"]
    model_class = config["model_class"]

    # Create tasks and payloads
    #slice_dict = json.loads(args.slice_dict) if args.slice_dict else {}
    #task_config.update({"slice_dict": slice_dict})
    task_config["active_slice_heads"] = active_slice_heads


    if args.model_type == 'naive':
        slice_names = []
    else:
        if not args.slices:
            raise ValueError('Need to provide a list of slices!')
        slice_names = args.slices
        #slice_names = [233,247,57]

    print('Using {} slices: {}'.format(len(slice_names), slice_names))

    tasks, payloads = create_birds_tasks_payloads(slice_names, X_splits, Y_splits, image_id_splits, attrs_dict, **task_config)

    print('tasks: ', tasks)
    print('payloads: ')
    pprint(payloads)
    # Create evaluation payload with test_slices -> primary task head
    #task_config.update({"slice_dict": slice_dict})
    task_config["active_slice_heads"] = {
        # turn pred labelsets on, and use model's value for ind head
        "pred": True,
        "ind": active_slice_heads.get("ind", False),
        'shared_pred' : active_slice_heads.get('shared_pred', False)
    }
    #compute baseline numbers for all slices for each comparison
    if args.model_type == 'naive':
        slice_tasks, slice_payloads = create_birds_tasks_payloads(slice_names, X_splits, Y_splits, image_id_splits, attrs_dict, **task_config)
    else: #just evaluate on the slices of interest
        slice_tasks, slice_payloads = create_birds_tasks_payloads(slice_names, X_splits, Y_splits, image_id_splits, attrs_dict, **task_config)
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

    # if active_slice_heads:
    #     tasks = convert_to_slicing_tasks(tasks)

    if args.model_type == "manual":
        slice_loss_mult = (
            json.loads(args.slice_loss_mult) if args.slice_loss_mult else {}
        )
        for task in tasks:
            if task.name in slice_loss_mult.keys():
                task.loss_multiplier = slice_loss_mult[task.name]
                print(
                    "Override {} loss multiplier with{}.".format(
                        task.name, slice_loss_mult[task.name]
                    )
                )

    # Initialize and train model
    print('slice_tasks: ')
    pprint(slice_tasks)
    print('slice payloads: ')
    pprint(slice_payloads)
    
    model = model_class(tasks, **model_config)

    #pretrained_naive_model_filepath = '/dfs/scratch1/saelig/slicing/metal/logs/2019_05_18/naive_BirdClassificationTask_16_50_40/model_checkpoint_30.006666666666668.pth'
    #pretrained_naive_model_filepath = '/dfs/scratch1/saelig/slicing/metal/logs/2019_05_19/naive_BirdClassificationTask_00_18_53/model_checkpoint_25.006666666666668.pth'
    if args.pretrained_model:
        print('Loading weights from pretrained naive model')
        pretrained_naive_model_filepath = args.pretrained_model
        model.load_weights(pretrained_naive_model_filepath)

    trainer = MultitaskTrainer(**trainer_config)

    # Write config files
    trainer._set_writer()
    trainer.writer.write_config(model_config, "model_config")
    trainer.writer.write_config(task_config, "task_config")

    # train model
    trainer.train_model(model, payloads)

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
        default=True,
        help="Whether to map eval main head on validation set during training",
    )

    parser.add_argument(
        "--slice_loss_mult",
        type=str,
        default=False,
        help="Slice loss multipliers that override the default ones (1/num_slices).",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=False,
        help="filepath to pretrained naive model",
    )

    parser.add_argument(
        '--slices',
        nargs='+',
        type=int,
        required=False,
        help='list of attr ids'
        )


    parser = add_flags_from_config(parser, trainer_defaults)
    parser = add_flags_from_config(parser, model_defaults)
    parser = add_flags_from_config(parser, task_defaults)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
