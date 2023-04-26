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
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
)


def f(x):
    sigma = np.sin(x) ** 2 + 0.01
    noise = np.random.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    return x * np.sin(x) + noise


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


def coverage_fraction(y, y_low, y_high):
    return np.mean(np.logical_and(y >= y_low, y <= y_high))


def test_quantile_regression_mse(input_data, model):
    X_train, X_test, y_train, y_test = input_data
    qr_out = model.predict(X_test).mean()
    reg = GradientBoostingRegressor(loss="squared_error")
    reg.fit(X_train, y_train)
    gbt_out = reg.predict(X_test)

    mse = np.sqrt(np.mean(np.square(y_test - qr_out)))
    basemse = np.sqrt(np.mean(np.square(y_test - gbt_out)))
    assert abs(mse - basemse) <= 0.2, "MSE must be kept to a reasonable level"


def test_quantile_regression_coverage(input_data, model):
    X_train, X_test, y_train, y_test = input_data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Compute the coverage of the quanties on the train set
    for quantile in train_predictions.quantiles:
        preds = train_predictions.results(quantile)
        fraction = coverage_fraction(y_train, y_low=-np.inf, y_high=preds)
        assert (
            np.abs(quantile - fraction) <= 0.2
        ), "Quantile predctions must be calibrated"

    # Compute the coverage of the quanties on the test set
    for quantile in test_predictions.quantiles:
        preds = test_predictions.results(quantile)
        fraction = coverage_fraction(y_test, y_low=-np.inf, y_high=preds)
        assert (
            np.abs(quantile - fraction) <= 0.2
        ), "Quantile predctions must be calibrated"
