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
from typing import Optional

import numpy as np
import pandas as pd

from syne_tune.optimizer.schedulers.searchers.bore.bore import Bore
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.model import Model
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges


class BOREModel(Model):
    def __init__(
        self,
        config_space,
        mode: str,
        random_state: Optional[np.random.RandomState] = None,
        **kwargs
    ):
        super(BOREModel, self).__init__(
            random_state=random_state, config_space=config_space, mode=mode
        )
        self.bore = Bore(
            config_space=config_space,
            metric="objective",
            mode=mode,
            random_seed=random_state.randint(2**32 - 1),
            **kwargs
        )

        self.bore._points_to_evaluate = None

    def suggest(self) -> dict:
        next = self.bore.get_config()
        return next

    def to_nd_array(self, X: pd.DataFrame):
        hp_range = make_hyperparameter_ranges(self.config_space)
        configs = [dict(zip(X.columns, v)) for v in X.values]
        return np.stack([hp_range.to_ndarray(config) for config in configs])

    def fit(self, df_features: pd.DataFrame, y: np.array):
        self.bore.inputs = self.to_nd_array(df_features)
        self.bore.targets = y
        self.bore._model_needs_training = True
