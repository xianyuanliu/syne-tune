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
from dataclasses import dataclass
from typing import Optional, List

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate_searcher import (
    SurrogateSearcher,
)
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)


@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    resource_attr: str
    points_to_evaluate: List[dict]
    max_t: Optional[int] = None
    max_resource_attr: Optional[str] = None
    use_surrogates: bool = False
    num_brackets: Optional[int] = 1
    verbose: Optional[bool] = False


class Methods:
    # single fidelity
    RS = "RS"
    GP = "GP"
    TPE = "TPE"
    BORE = "BORE"
    REA = "REA"
    HEBO = "HEBO"

    # multifidelity
    ASHA = "ASHA"
    BOHB = "BOHB"
    BOREHB = "BOREHB"
    MOBSTER = "MOB"
    HYPERTUNE = "HT"

    # Ours
    QR = "QR"
    CQR = "CQR"

    @staticmethod
    def multifidelity(method):
        return f"{method}-last"


def _max_resource_attr_or_max_t(
    args: MethodArguments, max_t_name: str = "max_t"
) -> dict:
    if args.max_resource_attr is not None:
        return {"max_resource_attr": args.max_resource_attr}
    else:
        assert args.max_t is not None
        return {max_t_name: args.max_t}


methods = {
    Methods.RS: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.GP: lambda method_arguments: FIFOScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options={"debug_log": False},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.REA: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher=RegularizedEvolution(
            config_space=method_arguments.config_space,
            metric=method_arguments.metric,
            mode=method_arguments.mode,
            random_seed=method_arguments.random_seed,
            points_to_evaluate=method_arguments.points_to_evaluate,
            population_size=10,
            sample_size=5,
        ),
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.BOHB: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.TPE: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.BORE: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
        search_options={"classifier": "mlp-sklearn"},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.BOREHB: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
        search_options={"classifier": "xgboost", "init_random": 10},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.MOBSTER: lambda method_arguments: HyperbandScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options=dict(
            debug_log=False,
            opt_skip_init_length=500,
            opt_skip_period=25,
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.HYPERTUNE: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="hypertune",
        search_options=dict(
            debug_log=False,
            model="gp_independent",
            opt_skip_init_length=500,
            opt_skip_period=25,
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
}


def make_conformal_searcher(
    surrogate: str,
    hyperband: bool = False,
    **surrogate_kwargs,
):
    # mapping from experiment name to SurrogateSearcher argument
    surrogate_naming_mapping = {
        Methods.QR: "QuantileRegression",
        Methods.CQR: "ConformalQuantileRegression",
        Methods.BORE: "BORE",
        Methods.REA: "REA",
        Methods.GP: "GP",
    }

    def _fun(method_arguments):

        if surrogate == Methods.QR:
            surrogate_kwargs.update(
                dict(
                    learning_rate=0.04,
                    max_depth=7,
                    n_estimators=200,
                    quantiles=5,
                )
            )
        searcher = SurrogateSearcher(
            metric=method_arguments.metric,
            mode=method_arguments.mode,
            config_space=method_arguments.config_space,
            surrogate=surrogate_naming_mapping[surrogate],
            random_seed=method_arguments.random_seed,
            max_fit_samples=400 if "GP" in surrogate else 1000,
            points_to_evaluate=method_arguments.points_to_evaluate,
            min_samples_to_conformalize=32 if "CQR" in surrogate else None,
            **surrogate_kwargs,
        )
        scheduler_args = dict(
            config_space=method_arguments.config_space,
            mode=method_arguments.mode,
            metric=method_arguments.metric,
            random_seed=method_arguments.random_seed,
            searcher=searcher,
        )
        if hyperband:
            return HyperbandScheduler(
                max_t=method_arguments.max_t,
                resource_attr=method_arguments.resource_attr,
                **scheduler_args,
            )
        else:
            return FIFOScheduler(**scheduler_args)

    return _fun


surrogates_methods = [
    Methods.QR,
    Methods.CQR,
    Methods.BORE,
    Methods.REA,
    Methods.GP,
]

# Add single fidelity version for all surrogate method
for surrogate in surrogates_methods:
    methods[surrogate] = make_conformal_searcher(
        surrogate,
        hyperband=False,
        update_frequency=1,
    )

# add multifidelity versions for all surrogate methods
for surrogate in surrogates_methods:
    methods[Methods.multifidelity(surrogate)] = make_conformal_searcher(
        surrogate,
        hyperband=True,
        update_frequency=1,
    )


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarking.nursery.benchmark_conformal.benchmark_definitions import (
        benchmark_definitions,
    )

    print(f"Checking initialization of {list(methods.keys())[::-1]}")
    # sys.exit(0)
    benchmarks = ["fcnet-protein", "nas201-cifar10", "lcbench-Fashion-MNIST"]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        points_to_evaluate = [
            {k: v.sample() for k, v in backend.blackbox.configuration_space.items()}
            for _ in range(4)
        ]
        print(f"Checking initialization of {list(methods.keys())[::-1]}")
        for method_name, method_fun in list(methods.items())[::-1]:
            print(f"checking initialization of: {method_name}, {benchmark_name}")
            # if method_name != Methods.QHB_XGB:
            #     continue

            scheduler = method_fun(
                MethodArguments(
                    config_space=backend.blackbox.configuration_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=0,
                    max_t=max(backend.blackbox.fidelity_values),
                    resource_attr=next(iter(backend.blackbox.fidelity_space.keys())),
                    use_surrogates=benchmark_name == "lcbench-Fashion-MNIST",
                    points_to_evaluate=points_to_evaluate,
                )
            )
            print(scheduler.suggest(0))
            print(scheduler.suggest(1))
