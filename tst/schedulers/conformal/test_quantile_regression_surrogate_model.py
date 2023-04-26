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
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from syne_tune.config_space import randint, uniform
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)


def f(x):
    sigma = np.sin(x) ** 2 + 0.01
    noise = np.random.normal(scale=sigma)
    return noise


@pytest.fixture
def input_data(
    nsamples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=nsamples)).T

    Y = f(X.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def model(input_data) -> GradientBoostingQuantileRegressor:
    X_train, X_test, y_train, y_test = input_data
    model = GradientBoostingQuantileRegressor(quantiles=9)
    model.fit(X_train, y_train)
    return model


def test_quantile_surrogate_model_base():
    n = 200
    config_space = {"x": randint(0, 10), "y": randint(0, 10)}
    X = pd.DataFrame(
        [{k: v.sample() for k, v in config_space.items()} for _ in range(n)]
    )
    y = X.values.sum(axis=1, keepdims=True)
    model = QuantileRegressionSurrogateModel(
        config_space=config_space,
        mode="min",
    )
    model.fit(X, y)

    sample = model._get_sampler(X)()
    assert sample.shape == (n,)

    mu = model.predict(X).mean()
    l1_error = np.abs(mu - y.reshape(-1)).mean()
    assert l1_error < 0.5

    median = model.predict(X).results(0.5)
    l1_error = np.abs(median - y.reshape(-1)).mean()
    assert l1_error < 0.5
