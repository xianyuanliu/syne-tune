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

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42

import pandas as pd
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import dill
import numpy as np
from matplotlib import pyplot as plt

from benchmarking.nursery.benchmark_conformal.results_analysis.critical_diagram import (
    draw_cd_diagram,
)
from benchmarking.nursery.benchmark_conformal.baselines import (
    Methods,
)
from benchmarking.nursery.benchmark_conformal.results_analysis.load_experiments_parallel import (
    load_benchmark_results,
)
from benchmarking.nursery.benchmark_conformal.results_analysis.method_styles import (
    method_styles,
    plot_range,
)
from benchmarking.nursery.benchmark_conformal.util import (
    figure_folder,
)
from syne_tune.util import catchtime

lw = 2.5
alpha = 0.7
matplotlib.rcParams.update({"font.size": 15})
benchmark_families = ["fcnet", "lcbench", "nas201", "yahpo"]
benchmark_names = {
    "fcnet": "\\FCNet{}",
    "nas201": "\\NASBench{}",
    "lcbench": "\\LCBench{}",
    "yahpo": "\\NASSurr{}",
}


def plot_result_benchmark(
    t_range: np.array,
    method_dict: Dict[str, np.array],
    title: str,
    rename_dict: dict,
    method_styles: Optional[Dict] = None,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = True,
):
    agg_results = {}

    if plot_regret:
        mode = "min"
        min_value = min([v.min() for v in method_dict.values()])
        max_value = max([v.max() for v in method_dict.values()])
        best_result, worse_result = (
            (min_value, max_value) if mode == "min" else (max_value, min_value)
        )

    if len(method_dict) > 0:
        if ax is None:
            fig, ax = plt.subplots()
        for algorithm in method_dict.keys():
            if methods_to_show is not None and algorithm not in methods_to_show:
                continue
            renamed_algorithm = rename_dict.get(algorithm, algorithm)
            if renamed_algorithm not in method_styles:
                method_style = method_styles.get(algorithm)
            else:
                method_style = method_styles.get(renamed_algorithm)
            if method_style is None:
                logging.warning(f"{algorithm} not found in method")
            # (num_seeds, num_time_steps)
            y_ranges = method_dict[algorithm]
            if plot_regret:
                y_ranges = (y_ranges - best_result) / (worse_result - best_result)
            mean = y_ranges.mean(axis=0)
            std = y_ranges.std(axis=0, ddof=1) / np.sqrt(y_ranges.shape[0])
            if method_style:
                ax.fill_between(
                    t_range,
                    mean - std,
                    mean + std,
                    color=method_style.color,
                    alpha=0.1,
                )
                ax.plot(
                    t_range,
                    mean,
                    color=method_style.color,
                    linestyle=method_style.linestyle,
                    marker=method_style.marker,
                    label=renamed_algorithm,
                )
                if plot_regret:
                    ax.set_yscale("log")
            else:
                ax.fill_between(
                    t_range,
                    mean - std,
                    mean + std,
                    alpha=0.1,
                )
                ax.plot(
                    t_range,
                    mean,
                    label=renamed_algorithm,
                    alpha=alpha,
                )

            agg_results[algorithm] = mean

        ax.set_xlabel("Wallclock time")
        ax.legend()
        ax.set_title(title)
    return ax


def plot_task_performance_over_time(
    benchmark_results: Dict[str, Tuple[np.array, Dict[str, np.array]]],
    rename_dict: dict,
    result_folder: Path,
    method_styles: Optional[Dict] = None,
    title: str = None,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = False,
):
    print(f"plot rank through time on {result_folder}")
    for benchmark, (t_range, method_dict) in benchmark_results.items():
        ax = plot_result_benchmark(
            t_range=t_range,
            method_dict=method_dict,
            title=benchmark,
            method_styles=method_styles,
            ax=ax,
            methods_to_show=methods_to_show,
            rename_dict=rename_dict,
            plot_regret=plot_regret,
        )
        ax.set_ylabel("objective")
        if title is not None:
            ax.set_title(title)
        if not plot_regret:
            if benchmark in plot_range:
                plotargs = plot_range[benchmark]
                ax.set_ylim([plotargs.ymin, plotargs.ymax])
                ax.set_xlim([plotargs.xmin, plotargs.xmax])

        if ax is not None:
            plt.tight_layout()
            filepath = result_folder / f"{benchmark}.pdf"
            plt.savefig(filepath)
        ax = None


def load_and_cache(
    experiment_tags: List[str],
    methods: Optional[List[str]] = None,
    load_cache_if_exists: bool = True,
    num_time_steps=100,
    max_seed=10,
    experiment_filter=None,
):
    result_file = Path(
        f"~/Downloads/cached-results-{str(experiment_tags[0])}.dill"
    ).expanduser()
    if load_cache_if_exists and result_file.exists():
        with catchtime(f"loading results from {result_file}"):
            with open(result_file, "rb") as f:
                benchmark_results = dill.load(f)
    else:
        print(f"regenerating results to {result_file}")
        with catchtime("load benchmark results"):
            benchmark_results = load_benchmark_results(
                experiment_tags=experiment_tags,
                methods=methods,
                num_time_steps=num_time_steps,
                max_seed=max_seed,
                experiment_filter=experiment_filter,
            )

        with open(result_file, "wb") as f:
            dill.dump(benchmark_results, f)

    return benchmark_results


def plot_ranks(
    ranks,
    benchmark_results,
    title: str,
    rename_dict: dict,
    result_folder: Path,
    methods_to_show: List[str],
):
    plt.figure()
    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    ys = ranks.reshape(benchmark_results.shape).mean(axis=(1, 2))
    xs = np.linspace(0, 1, ys.shape[-1])
    for i, method in enumerate(methods_to_show):
        renamed_algorithm = rename_dict.get(method, method)
        if renamed_algorithm not in method_styles:
            method_style = method_styles.get(method)
        else:
            method_style = method_styles.get(renamed_algorithm)

        if method_style is None:
            logging.warning(f"{method} not found in method")
            kwargs = {}
        else:
            kwargs = method_style.plot_kwargs()
        plt.plot(
            xs,
            ys[i],
            label=rename_dict.get(method, method),
            alpha=alpha,
            lw=lw,
            **kwargs,
        )
    plt.xlabel("% Budget Used")
    plt.ylabel("Method rank")
    plt.xlim(0, 1)
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(result_folder / f"{title}.pdf")


def stack_benchmark_results(
    benchmark_results_dict: Dict[str, Tuple[np.array, Dict[str, np.array]]],
    methods_to_show: Optional[List[str]],
    benchmark_families: List[str],
) -> Dict[str, np.array]:
    """
    Stack benchmark results between benchmarks of the same family.
    :param benchmark_results_dict:
    :param methods_to_show:
    :return: dictionary from benchmark family to tensor results with shape
    (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    """
    for benchmark, (t_range, method_dict) in benchmark_results_dict.items():
        for method in methods_to_show:
            if method not in method_dict:
                print(
                    f"removing method {method} from methods to show as it is not present in all benchmarks"
                )
                methods_to_show.remove(method)

    res = {}
    for benchmark_family in benchmark_families:
        # list of the benchmark of the current family
        benchmarks_family = [
            benchmark
            for benchmark in benchmark_results_dict.keys()
            if benchmark_family in benchmark
        ]

        benchmark_results = []
        for benchmark in benchmarks_family:
            benchmark_result = [
                benchmark_results_dict[benchmark][1][method]
                for method in methods_to_show
            ]
            benchmark_result = np.stack(benchmark_result)
            benchmark_results.append(benchmark_result)

        # (num_benchmarks, num_methods, num_min_seeds, num_time_steps)
        benchmark_results = np.stack(benchmark_results)

        if benchmark_family in ["lcbench", "yahpo"]:
            # max instead of minimization, todo pass the mode somehow
            benchmark_results *= -1

        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        res[benchmark_family] = benchmark_results.swapaxes(0, 1)

    return res


def plot_critical_diagram(
    ranks,
    benchmark_family: str,
    methods_to_show: List[str],
    result_folder,
    rename_dict=None,
):
    """
    :param ranks: shape (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    :param benchmark_family:
    :param methods_to_show:
    :param rename_dict:
    :return:
    """
    # averages over seeds and time-steps (num_methods, num_benchmarks)
    df_ranks = pd.DataFrame(
        ranks.mean(axis=(2,)).reshape(-1),
        columns=["rank"],
    )
    if rename_dict:
        methods_to_show_renamed = [rename_dict.get(s, s) for s in methods_to_show]
    else:
        methods_to_show_renamed = methods_to_show
    col_methods = [
        [m] * (len(df_ranks) // len(methods_to_show)) for m in methods_to_show_renamed
    ]
    df_ranks["scheduler"] = [x for m in col_methods for x in m]

    try:
        draw_cd_diagram(
            df=df_ranks,
            method_column="scheduler",
            rank_column="rank",
            title=benchmark_family,
            folder=str(result_folder),
        )
    except Exception as e:
        print(str(e))
        traceback.print_stack()


def generate_rank_results(
    benchmark_families: List[str],
    stacked_benchmark_results: Dict[str, np.array],
    methods_to_show: Optional[List[str]],
    rename_dict: dict,
    result_folder: Path,
):
    rows = []
    for benchmark_family in benchmark_families:
        print(benchmark_family)
        # list of the benchmark of the current family
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        benchmark_results = stacked_benchmark_results[benchmark_family]

        ranks = pd.DataFrame(
            benchmark_results.reshape(len(benchmark_results), -1)
        ).rank()
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        ranks = ranks.values.reshape(benchmark_results.shape)
        # (num_methods, num_benchmarks)
        avg_ranks_per_tasks = ranks.mean(axis=(2, 3))
        for i in range(benchmark_results.shape[1]):
            row = {"benchmark": f"{benchmark_family}-{i}"}
            row.update(dict(zip(methods_to_show, avg_ranks_per_tasks[:, i])))
            rows.append(row)

        plot_ranks(
            ranks,
            benchmark_results,
            benchmark_family,
            rename_dict,
            result_folder,
            methods_to_show,
        )

    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    all_results = np.concatenate(list(stacked_benchmark_results.values()), axis=1)
    all_ranks = pd.DataFrame(all_results.reshape(len(all_results), -1)).rank()
    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    # all_ranks = all_ranks.values.reshape(all_results.shape)
    plot_ranks(
        all_ranks.values,
        all_results,
        "Average-rank",
        rename_dict,
        result_folder,
        methods_to_show,
    )


def generate_critical_diagrams(
    benchmark_families: List[str],
    stacked_benchmark_results: Dict[str, np.array],
    methods_to_show: Optional[List[str]],
    rename_dict,
    result_folder,
):
    all_ranks = []
    for benchmark_family in benchmark_families:
        benchmark_results = stacked_benchmark_results[benchmark_family]

        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        ranks = (
            pd.DataFrame(benchmark_results.reshape(len(benchmark_results), -1))
            .rank()
            .values.reshape(benchmark_results.shape)
        )
        all_ranks.append(ranks)
    all_ranks = np.concatenate(all_ranks, axis=1)
    for budget_ratio in [0.25, 0.5, 0.75, 1.0]:
        num_time_steps = benchmark_results.shape[-1]
        t = min(int(num_time_steps * budget_ratio), num_time_steps - 1)

        df_perf = (
            pd.DataFrame(
                all_ranks[..., t].reshape(len(benchmark_results), -1),
                index=[rename_dict.get(x, x) for x in methods_to_show],
            )
            .unstack()
            .reset_index()
        )
        df_perf.columns = ["benchmark", "method", "rank"]

        try:
            draw_cd_diagram(
                df=df_perf,
                method_column="method",
                rank_column="rank",
                title=f"Rank@{int(budget_ratio*100)}%",
                folder=result_folder,
            )
        except Exception as e:
            print(str(e))


def plot_average_normalized_regret(
    stacked_benchmark_results,
    rename_dict: dict,
    result_folder: Path,
    method_styles: Optional[Dict] = None,
    title: str = None,
    show_ci: bool = False,
    ax=None,
    methods_to_show: list = None,
):
    normalized_regrets = []
    for benchmark_family in benchmark_families:
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        benchmark_results = stacked_benchmark_results[benchmark_family]
        # uncomment to remove outliers
        # benchmark_results = np.clip(benchmark_results, a_min=None, a_max=np.percentile(benchmark_results, 99))
        benchmark_results_best = benchmark_results.min(axis=(0, 2, 3), keepdims=True)
        benchmark_results_worse = benchmark_results.max(axis=(0, 2, 3), keepdims=True)
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        normalized_regret = (benchmark_results - benchmark_results_best) / (
            benchmark_results_worse - benchmark_results_best
        )
        normalized_regrets.append(normalized_regret)

    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    normalized_regrets = np.concatenate(normalized_regrets, axis=1)

    # (num_methods, num_time_steps)
    avg_regret = normalized_regrets.mean(axis=(1, 2))
    std_regret = normalized_regrets.std(axis=2).mean(axis=1) if show_ci else None

    if ax is None:
        fig, ax = plt.subplots()
    for i, algorithm in enumerate(methods_to_show):
        renamed_algorithm = rename_dict.get(algorithm, algorithm)
        if renamed_algorithm not in method_styles:
            method_style = method_styles.get(algorithm)
        else:
            method_style = method_styles.get(renamed_algorithm)
        if method_style is None:
            logging.warning(f"{algorithm} not found in method")

        # (num_seeds, num_time_steps)
        mean = avg_regret[i]
        if method_style:
            ax.plot(
                np.arange(len(mean)) / len(mean),
                mean,
                color=method_style.color,
                linestyle=method_style.linestyle,
                marker=method_style.marker,
                label=renamed_algorithm,
                lw=lw,
                alpha=alpha,
            )
            if show_ci:
                std = std_regret[i]
                ax.fill_between(
                    np.arange(len(mean)) / len(mean),
                    mean - std,
                    mean + std,
                    color=method_style.color,
                    alpha=0.1,
                )
            ax.set_yscale("log")

    plt.xlabel("% Budget Used")
    ax.set_ylabel("Average normalized regret")
    plt.xlim(0, 1)
    plt.ylim(6e-3, None)
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(result_folder / f"{title}.pdf")
    plt.show()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=True,
        help="the experiment tag that was displayed when running the experiment",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        required=False,
    )

    single_fidelity = [
        Methods.RS,
        Methods.TPE,
        Methods.REA,
        Methods.GP,
        Methods.BORE,
        Methods.CQR,
    ]
    multi_fidelity = [
        Methods.ASHA,
        Methods.BOHB,
        Methods.MOBSTER,
        Methods.HYPERTUNE,
        Methods.multifidelity(Methods.CQR),
    ]

    multi_fidelity_last = [
        Methods.ASHA,
        Methods.MOBSTER,
        Methods.multifidelity(Methods.REA),
        Methods.multifidelity(Methods.GP),
        Methods.multifidelity(Methods.BORE),
        Methods.multifidelity(Methods.QR),
        Methods.multifidelity(Methods.CQR),
    ]

    methods_to_show = single_fidelity + multi_fidelity + multi_fidelity_last

    groups = {
        "single-fidelity": single_fidelity,
        "multi-fidelity": multi_fidelity,
        "multi-fidelity-extension": multi_fidelity_last,
        "all": single_fidelity + multi_fidelity,
    }

    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag

    print(args.__dict__)

    max_seed = args.max_seed
    num_time_steps = 50

    rename_dict = {
        Methods.multifidelity(Methods.REA): f"{Methods.REA}+MF",
        Methods.multifidelity(Methods.GP): f"{Methods.GP}+MF",
        Methods.multifidelity(Methods.BORE): f"{Methods.BORE}+MF",
        Methods.multifidelity(Methods.QR): f"{Methods.QR}+MF",
        Methods.multifidelity(Methods.CQR): f"{Methods.CQR}+MF",
    }

    with catchtime("load benchmark results"):
        benchmark_results = load_and_cache(
            experiment_tags=[experiment_tag],
            load_cache_if_exists=args.reuse_cache,
            max_seed=max_seed,
            num_time_steps=num_time_steps,
            methods=methods_to_show,
        )

    for group_name, methods in groups.items():
        folder_name = experiment_tag
        result_folder = figure_folder(Path("tabular") / folder_name / group_name)
        result_folder.mkdir(parents=True, exist_ok=True)

        stacked_benchmark_results = stack_benchmark_results(
            benchmark_results_dict=benchmark_results,
            methods_to_show=methods,
            benchmark_families=benchmark_families,
        )

        with catchtime("generating rank table"):
            generate_rank_results(
                stacked_benchmark_results=stacked_benchmark_results,
                benchmark_families=benchmark_families,
                methods_to_show=methods,
                rename_dict=rename_dict,
                result_folder=result_folder,
            )

        generate_critical_diagrams(
            stacked_benchmark_results=stacked_benchmark_results,
            benchmark_families=benchmark_families,
            methods_to_show=methods,
            result_folder=result_folder,
            rename_dict=rename_dict,
        )

        with catchtime("generating plots per task"):
            plot_task_performance_over_time(
                benchmark_results=benchmark_results,
                method_styles=method_styles,
                methods_to_show=methods,
                rename_dict=rename_dict,
                result_folder=result_folder,
            )

        plot_average_normalized_regret(
            stacked_benchmark_results=stacked_benchmark_results,
            method_styles=method_styles,
            methods_to_show=methods,
            rename_dict=rename_dict,
            result_folder=result_folder,
            title="Normalized-regret",
        )
