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
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
    QuantileRegressor,
    QuantileRegressorPredictions,
)


@dataclass
class ConformalQuantileCorrection:
    alpha: float
    sign: float = None
    correction: float = None

    def __post_init__(self):
        if self.alpha == 0.5:
            self.sign = 0.0
        elif self.alpha < 0.5:
            self.sign = 1.0
        elif self.alpha > 0.5:
            self.sign = -1.0
        else:
            raise RuntimeError(
                "Incorrect alpha provided to ConformalizedGradientBoostingQuantileRegressor"
            )


class ConformalizedGradientBoostingQuantileRegressor(GradientBoostingQuantileRegressor):
    conformal_correction: Dict[float, ConformalQuantileCorrection] = None

    def __init__(
        self,
        quantiles: Union[int, List[float]] = 9,
        valid_fraction: float = 0.10,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(quantiles, verbose, **kwargs)
        self.valid_fraction = valid_fraction
        self.conformal_correction = {
            alpha: ConformalQuantileCorrection(alpha) for alpha in self.quantiles
        }

    def fit(self, df_features: np.ndarray, y: np.array, **kwargs):
        x_training, x_validation, y_training, y_validation = train_test_split(
            df_features, y, test_size=self.valid_fraction
        )
        for quantile in tqdm(
            self.quantile_regressors,
            desc="Training Quantile Regression",
            disable=not self.verbose,
        ):
            self.quantile_regressors[quantile].fit(x_training, np.ravel(y_training))

        for alpha, cq in self.conformal_correction.items():
            residuals = cq.sign * (
                self.quantile_regressors[alpha].predict(x_validation).ravel()
                - y_validation.ravel()
            )

            # Compute the quantile loss - eq (1) in the paper
            if alpha < 0.5:
                target_quantile = 1 - alpha
            else:
                target_quantile = alpha

            cq.correction = np.quantile(residuals, q=target_quantile)  # using np.quantile for quantile of the residuals

    def predict(self, df_test: pd.DataFrame) -> QuantileRegressorPredictions:
        quantile_res = {
            quantile: regressor.predict(df_test)
            - self.conformal_correction[quantile].correction
            for quantile, regressor in self.quantile_regressors.items()
        }
        return QuantileRegressorPredictions.from_quantile_results(
            quantiles=self.quantiles,
            results=quantile_res,
        )
