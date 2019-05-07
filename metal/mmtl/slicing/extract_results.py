"""
Scans a directory for results for a particular task name and prints aggregate results.
    * Requires that all models in directory for the same task
    * Aggregate metrics will be computed over all files, so please separate models (naive, soft_param)

python extract_results.py --run_dir /dfs/scratch0/vschen/metal-mmtl/logs/2019_05_04 --task STSB
"""

import argparse
import json
import os
import pprint
import re
from collections import defaultdict

import numpy as np

metadata_to_extract = {"model_config.json": ["seed"], "config.json": ["l2", "lr"]}

metrics_to_extract = {"STSB": [r".*pearson_spearman$"]}


def recursive_find_key(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            item = recursive_find_key(v, key)
            if item is not None:
                return item


def extract_from_dict(source, target_keys, use_regex=False, return_keys=False):
    if use_regex:
        found_keys = []
        for source_key in source.keys():
            for target_k in target_keys:
                pattern = re.compile(target_k)
                if pattern.match(source_key):
                    found_keys.append(source_key)
        target_keys = found_keys

    results = {}
    for target_k in target_keys:
        results[target_k] = recursive_find_key(source, target_k)

    if return_keys:
        return results, target_keys
    else:
        return results


def main(args):
    run_dir = args.run_dir
    task_name = args.task

    results = {}  # {abs_dir: {"metadata": {...}, "metrics": {}}}

    # keep track of metrics we find to compute aggregate metrics
    all_metric_keys = set()

    for run_name in os.listdir(run_dir):
        abs_dir = os.path.join(run_dir, run_name)

        # extract desired metadata
        metadata = {}
        for config_fn, keys in metadata_to_extract.items():
            config_path = os.path.join(abs_dir, config_fn)
            if not os.path.isfile(config_path):
                print(f"Could not find {config_path}")
                continue
            config = json.load(open(config_path, "r"))
            metadata.update(extract_from_dict(config, keys))

        # extract appropriate metrics
        metrics_path = os.path.join(abs_dir, "slice_metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"Could not find {metrics_path}")
            continue

        slice_metrics = json.load(open(metrics_path, "r"))
        selected_results, metric_keys = extract_from_dict(
            slice_metrics,
            metrics_to_extract[task_name],
            use_regex=True,
            return_keys=True,
        )

        # update running list of metrics
        all_metric_keys = all_metric_keys.union(set(metric_keys))

        # store all values in results
        results[abs_dir] = {"metrics": selected_results, "metadata": metadata}

    aggregate_metrics = defaultdict(list)
    for dirname, values in results.items():
        print("*" * 10, dirname, "*" * 10)
        pprint.pprint(values["metadata"])
        pprint.pprint(values["metrics"])
        for k, v in values["metrics"].items():
            aggregate_metrics[k].append(v)

    print("*" * 80)
    print("Aggregate metrics")
    print("*" * 80)
    print("metric, avg, std, num_samples")
    for k, v in aggregate_metrics.items():
        print(f"{k}, {np.mean(v):.4f}, {np.std(v):.4f}, {len(v)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    main(args)
