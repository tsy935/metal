import argparse
import copy
import json
import numpy as np

from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads, task_defaults
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.slicing.slice_model import SliceModel
from metal.mmtl.slicing.slicing_tasks import convert_to_slicing_tasks
from metal.utils import add_flags_from_config, recursive_merge_dicts
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults

"""
python compare_to_baseline.py --seed 1 --tasks RTE --slice_dict '{"RTE": ["dash_semicolon", "more_people", "BASE"]}'
"""

def get_baseline_scores(tasks, seed, eval_payload, slice_dict):
    # Baseline Model
    task_name = tasks[0].name
    model = MetalModel(tasks, seed=seed, verbose=False)
    trainer = MultitaskTrainer(seed=seed)
    trainer.train_model(
        model,
        payloads,
        checkpoint_metric="{}/{}_valid/{}_gold/accuracy".format(task_name, task_name, task_name),
        checkpoint_metric_mode="max",
        checkoint_best=True,
        writer="tensorboard",
        optimizer="adamax",
        lr=5e-5,
        l2=1e-3,
        log_every=0.1,
        score_every=0.1,
        n_epochs=10,
        progress_bar=True,
        checkpoint_best=True,
        checkpoint_cleanup=False,
    )
    # NOTE: we need to retarget slices to the original RTE head
    for label_name in slice_dict[task_name].values():
        label_name = '{}_slice:'.format(task_name) + label_name
        eval_payload.retarget_labelset(label_name, task_name)
    return model.score(eval_payload)


def get_model_scores(tasks_slice, payloads_slice, seed, eval_payload, slice_dict):
    task_name = tasks_slice[0].name
    slicing_tasks = convert_to_slicing_tasks(tasks_slice)
    slice_model = SliceModel(slicing_tasks, seed=seed, verbose=False)
    task_metrics = []
    prefix = '{}/{}_'.format(task_name, task_name)
    for p_name, metric in [('train', 'loss'), ('valid', 'accuracy')]:
        task_metrics += [prefix+p_name+'/{}_gold/{}'.format(task_name, metric)]
        for label_name in slice_dict[task_name].values():
            task_metrics += [prefix + p_name + '/{}_slice:{}/{}'.format(task_name, label_name, metric)]
    print(task_metrics)
    trainer = MultitaskTrainer(seed=seed)
    trainer.train_model(
        slice_model,
        payloads_slice,
        task_metrics=task_metrics,
        checkpoint_metric="{}/{}_valid/{}_gold/accuracy".format(task_name, task_name, task_name),
        checkpoint_metric_mode="max",
        checkoint_best=True,
        writer="tensorboard",
        optimizer="adamax",
        lr=1e-5,
        l2=1e-3,
        log_every=0.1,
        score_every=0.1,
        n_epochs=20,
        progress_bar=True,
        checkpoint_best=True,
        checkpoint_cleanup=False
    )
    return slice_model.score(eval_payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare specified model with MetalModel on single tasks", add_help=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=np.random.randint(1e6),
        help="A single seed to use for trainer, model, and task configs",)
    parser.add_argument(
        "--model_type",
        type=str,
        default="metal",
        help="Baseline model type")
    parser = add_flags_from_config(parser, trainer_defaults)
    parser = add_flags_from_config(parser, model_defaults)
    parser = add_flags_from_config(parser, task_defaults)
    args = parser.parse_args()

    # Extract flags into their respective config files
    trainer_config = recursive_merge_dicts(
        trainer_defaults, vars(args), misses="ignore"
    )
    model_config = recursive_merge_dicts(model_defaults, vars(args), misses="ignore")
    task_config = recursive_merge_dicts(task_defaults, vars(args), misses="ignore")
    args = parser.parse_args()
    task_kwargs = {
        "dl_kwargs": {"batch_size": 16},
        "freeze_bert": False,
        "bert_model": 'bert-base-cased',
        "max_len": 200,
        "attention": False
    }
    task_names = args.tasks.split(',')  # ['RTE']
    assert len(task_names) == 1

    # Create tasks and payloads
    tasks, payloads = create_glue_tasks_payloads(task_names, **task_kwargs)
    slice_dict = json.loads(args.slice_dict)
    task_kwargs.update({"slice_dict": slice_dict})
    task_kwargs['attention'] = None
    tasks_slice, payloads_slice = create_glue_tasks_payloads(
        task_names, **task_kwargs
    )

    eval_payload = copy.deepcopy(payloads_slice[1])

    if args.model_type.upper() == 'METAL':
        baseline_scores = get_baseline_scores(
            tasks, args.seed, eval_payload, slice_dict
        )
    else:
        raise Exception('Not Yet Implemented!')

    eval_payload = copy.deepcopy(payloads_slice[1])
    model_scores = get_model_scores(
        tasks_slice, payloads_slice, args.seed, eval_payload, slice_dict
    )
    print(baseline_scores)
    print(model_scores)
