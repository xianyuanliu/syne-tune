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
import pytest

from benchmarking.nursery.benchmark_conformal.baselines import (
    MethodArguments,
    methods,
    Methods,
)
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import uniform

config_space = {"x": uniform(0, 1)}
metric = "metric"
method_arguments = MethodArguments(
    config_space=config_space,
    metric=metric,
    mode="max",
    random_seed=0,
    max_t=100,
    resource_attr="epoch",
    points_to_evaluate=None,
)


@pytest.mark.parametrize(
    "method",
    [
        Methods.QR,
        Methods.CQR,
        Methods.multifidelity(Methods.QR),
        Methods.multifidelity(Methods.CQR),
        Methods.multifidelity(Methods.BORE),
        Methods.multifidelity(Methods.REA),
    ],
)
def test_baseline(method: str):
    method_fun = methods[method]
    scheduler = method_fun(method_arguments)
    print(scheduler.suggest(0))
    print(scheduler.suggest(1))


@pytest.mark.parametrize(
    "method,surrogate_name,min_samples_to_conformalize",
    [
        (Methods.QR, "QuantileRegression", None),
        (Methods.CQR, "ConformalQuantileRegression", 32),
        (Methods.multifidelity(Methods.QR), "QuantileRegression", None),
        (Methods.multifidelity(Methods.CQR), "ConformalQuantileRegression", 32),
        (Methods.multifidelity(Methods.REA), "REA", None),
        (Methods.multifidelity(Methods.BORE), "BORE", None),
    ],
)
def test_baseline_parametrization(method, surrogate_name, min_samples_to_conformalize):
    method_fun = methods[method]
    scheduler = method_fun(method_arguments)
    searcher = scheduler.searcher
    assert searcher.surrogate == surrogate_name
    for i in range(10):
        print(scheduler.suggest(i))
    for i in range(10):
        scheduler.on_trial_result(
            Trial(trial_id=i, config={}, creation_time=None),
            result={"epoch": 1, metric: i},
        )
    scheduler.suggest(10)
    assert searcher.surrogate_model is not None
    if surrogate_name in ["QuantileRegression", "ConformalQuantileRegression"]:
        if surrogate_name == "QuantileRegression":
            assert not hasattr(
                searcher.surrogate_model.quantile_regressor,
                "min_samples_to_conformalize",
            )
            assert searcher.surrogate_model.quantile_regressor.valid_fraction == 0.0
        else:
            assert searcher.surrogate_model.quantile_regressor.valid_fraction == 0.1
            assert (
                searcher.surrogate_model.quantile_regressor.min_samples_to_conformalize
                == min_samples_to_conformalize
            )
