"""
Sample:
python search.py --max_search 2 --search_metric RTE/RTE_test/RTE_gold/accuracy --config /dfs/scratch0/vschen/metal-mmtl/metal/mmtl/slicing/sample_RTE_config.json
"""

import argparse
import json
import os
import subprocess

from metal.mmtl.slicing.launch import get_parser, main as launch
from metal.tuners.random_tuner import RandomSearchTuner


def main(args):
    search_config = json.load(open(args.config, "r"))
    assert "search_space" in search_config and "fixed_args" in search_config
    search_space = search_config["search_space"]
    fixed_args = search_config["fixed_args"]

    tuner = RandomSearchTuner(None, seed=123)
    configs = tuner.config_generator(search_space, args.max_search, tuner.rng, True)

    config_to_metrics = {}
    for search_conf in configs:
        full_conf = {}
        full_conf.update(search_conf)
        full_conf.update(fixed_args)

        arg_list = []
        for k, v in full_conf.items():
            # make sure the double quotes are correctly formatted
            if isinstance(v, dict):
                v = json.dumps(v)
            arg_list.extend([f"--{k}", str(v)])

        # print command being run
        print("*" * 80)
        print("python launch.py", " ".join(arg_list))
        print("*" * 80)
        parser = get_parser()
        launch_args = parser.parse_args(arg_list)
        metrics_path = launch(launch_args)
        metrics_dict = json.load(open(metrics_path, "r"))
        config_to_metrics[(metrics_path, str(search_conf))] = metrics_dict

    best_path_config = None
    best_metric = None
    if args.search_metric:
        metric_fns = {"max": max, "min": min}
        metric_cmp = metric_fns[args.search_metric_mode]

    for path_config, metrics in config_to_metrics.items():
        print("Searching path:", path_config)
        print("Metrics:", metrics)
        if args.search_metric:
            # no metrics set, pick the current one
            if not best_metric:
                best_path_config = path_config
                best_metric = metrics[args.search_metric]
            else:
                # compare to previous metrics
                curr_metric = metrics[args.search_metric]
                if metric_cmp(curr_metric, best_metric) == curr_metric:
                    best_path_config = path_config
                    best_metric = curr_metric

    best_path, best_config = best_path_config
    print("*" * 80)
    print("Best path:", best_path)
    print("Best config:", best_config)
    print("Best metric:", best_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_search",
        type=int,
        default=None,
        help=(
            "Number of iterations to search (see tuner.config_generator). "
            "If None, searches all discrete combinations."
        ),
    )
    parser.add_argument(
        "--search_metric",
        type=str,
        default=None,
        help="Metric from metrics_path that we should optimize for",
    )
    parser.add_argument(
        "--search_metric_mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Do we want to save the max or min of this search metric?",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config .json with fixed_args and search_space fields",
    )
    args = parser.parse_args()
    main(args)
