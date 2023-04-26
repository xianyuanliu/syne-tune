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
from functools import partial

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.model import Model
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.bore_model import (
    BOREModel,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.rea_searcher import (
    REASearcher,
)

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.gp_searcher import (
    GPSearcher,
)

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)


def surrogate_cls_from_string(surrogate_name: str) -> Model:
    if surrogate_name == "BORE":
        return BOREModel
    elif surrogate_name == "REA":
        return REASearcher
    elif surrogate_name == "GP":
        return GPSearcher
    elif surrogate_name == "QuantileRegression":
        return partial(
            QuantileRegressionSurrogateModel,
            quantiles=5,
            min_samples_to_conformalize=None,
        )
    elif surrogate_name == "ConformalQuantileRegression":
        return partial(
            QuantileRegressionSurrogateModel,
            min_samples_to_conformalize=32,
            valid_fraction=0.1,
        )

    raise ValueError(surrogate_name)
