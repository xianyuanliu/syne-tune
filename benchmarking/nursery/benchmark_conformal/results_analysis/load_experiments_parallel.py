# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import get_metadata
from tqdm import tqdm
from collections import defaultdict
from syne_tune.util import catchtime
from pathlib import Path
import os
from joblib import Parallel, delayed


def name_filter_from_description(experiment_tags, methods):
    def name_filter(name):
        for tag in experiment_tags:
            if tag in name:
                for method in methods:
                    if method in name:
                        return True
        return False

    return name_filter


def load_result(name, metadata):
    usecols = [metadata["metric_names"][0], "st_tuner_time", "trial_id", "st_decision"]
    try:
        return pd.read_csv(
            Path(metadata["path"]) / name / "results.csv.zip", usecols=usecols
        )
    except Exception:
        return None


def convert_to_numpy(benchmark_df, num_time_steps: int = 20):
    t_min = 0
    # the last time step is the median of the stopping time of all algorithms
    t_max = benchmark_df.groupby(["algorithm"]).max().st_tuner_time.median()
    t_range = np.linspace(t_min, t_max, num_time_steps)
    seed_results = {}
    for algorithm in benchmark_df["algorithm"].unique():
        ts = []
        ys = []
        df_scheduler = benchmark_df[benchmark_df["algorithm"] == algorithm]
        for seed in sorted(df_scheduler["seed"].unique()):
            sub_df = df_scheduler[df_scheduler["seed"] == seed]
            ts.append(sub_df.loc[:, ST_TUNER_TIME].values)
            ys.append(sub_df.loc[:, "best"].values)

        # for each seed, find the best value at each regularly spaced time-step
        y_ranges = []
        for t, y in zip(ts, ys):
            indices = np.searchsorted(t, t_range, side="left")
            y_range = y[np.clip(indices, 0, len(y) - 1)]
            y_ranges.append(y_range)

        # (num_seeds, num_time_steps)
        y_ranges = np.stack(y_ranges)
        seed_results[algorithm] = y_ranges
    return t_range, seed_results


def compute_best(dfs, metadatas):
    benchmark_dfs = defaultdict(list)
    for df_bench, metadata in tqdm(list(zip(dfs, metadatas.values()))):
        if df_bench is None:
            continue
        benchmark = metadata["benchmark"]
        metric_name = metadata["metric_names"][0]
        for key in ["algorithm", "seed"]:
            df_bench[key] = metadata[key]
        if metadata["metric_mode"] == "min":
            df_bench["best"] = df_bench[metric_name].cummin()
        else:
            df_bench["best"] = df_bench[metric_name].cummax()
            # todo perhaps rename column in O(1) for convenience
        if "lcbench" in benchmark or "nas301" in benchmark:
            assert metadata["metric_mode"] == "max", metadata
        else:
            assert metadata["metric_mode"] == "min"
        benchmark_dfs[benchmark].append(df_bench)

    benchmark_dfs = {
        k: pd.concat(v, ignore_index=True) for k, v in benchmark_dfs.items()
    }
    return benchmark_dfs


def show_number_seeds(benchmark_dfs):
    df_seeds = pd.DataFrame()
    for benchmark, df in benchmark_dfs.items():
        df_seeds[benchmark] = (
            benchmark_dfs[benchmark]
            .drop_duplicates(["algorithm", "seed"])
            .groupby(["algorithm"])
            .count()["trial_id"]
        )
    print("number of seeds available:")
    print(df_seeds.to_string())


def convert_all_to_numpy(benchmark_dfs, num_time_steps, max_seed):
    num_cores = max(1, os.cpu_count() // 2)

    benchmarks_numpy = Parallel(min(4, num_cores))(
        delayed(convert_to_numpy)(benchmark_df, num_time_steps=num_time_steps)
        for benchmark_df in benchmark_dfs.values()
    )

    min_num_seeds = min(
        values.shape[0] for x in benchmarks_numpy for method, values in x[1].items()
    )
    if min_num_seeds < max_seed:
        logging.warning(
            f"some methods have only {min_num_seeds} instead of the {max_seed} seeds asked, slicing all results "
            f"to {min_num_seeds} seeds."
        )
        num_seeds = {
            method: values.shape[0]
            for x in benchmarks_numpy
            for method, values in x[1].items()
        }
        logging.warning(num_seeds)
        benchmarks_numpy = [
            (
                t_range,
                {
                    method: values[:min_num_seeds]
                    for method, values in method_values.items()
                },
            )
            for (t_range, method_values) in benchmarks_numpy
        ]
    return dict(zip(benchmark_dfs.keys(), benchmarks_numpy))


def load_benchmark_results(
    experiment_tags,
    methods,
    num_time_steps: int = 20,
    max_seed: int = None,
    experiment_filter=None,
) -> Dict[str, Tuple[np.array, Dict[str, np.array]]]:
    """
    :param method_descriptions: list of method, tag filter, for instance [("ASHA", "tag1"), (None, "tag2)]
    will load together all runs from ASHA with tag1 and all runs from tag2
    :param max_seed: maximum seed to load, default to None to load all seeds
    :return:
    """
    name_filter = name_filter_from_description(experiment_tags, methods)

    with catchtime("Load metadata"):
        metadatas = get_metadata(name_filter)
    # todo strict metadata filtering as the one above may fail
    methods = set(methods)
    metadatas = {
        k: v
        for k, v in metadatas.items()
        if (max_seed is None or v["seed"] < max_seed) and v["algorithm"] in methods
    }
    if experiment_filter:
        metadatas = {k: v for k, v in metadatas.items() if experiment_filter(v)}
    print(f"loaded {len(metadatas)} experiment metadata")
    # metadatas = {k: v for k, v in metadatas.items() if "yahpo" not in v["benchmark"]}

    with catchtime("Load results dataframes"):
        # load results in parallel
        num_cores = max(1, os.cpu_count() // 2)
        dfs = Parallel(num_cores)(
            delayed(load_result)(name, metadata)
            for name, metadata in list(metadatas.items())
        )
    with catchtime("Compute best result over time"):
        benchmark_dfs = compute_best(dfs, metadatas)

    show_number_seeds(benchmark_dfs)

    with catchtime("Convert to numpy (num_seeds, num_time_steps)"):
        benchmark_results = convert_all_to_numpy(
            benchmark_dfs, num_time_steps, max_seed
        )
    return benchmark_results
