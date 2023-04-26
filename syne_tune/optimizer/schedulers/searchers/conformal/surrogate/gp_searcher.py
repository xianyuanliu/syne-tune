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

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.model import Model
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler


class GPSearcher(Model):
    def __init__(
        self,
        config_space,
        mode: str,
        metric: str,
        random_state: Optional[np.random.RandomState] = None,
        **kwargs
    ):
        super(GPSearcher, self).__init__(
            random_state=random_state, config_space=config_space, mode=mode
        )
        self.metric = metric
        self.model = None
        self.trial_id = -1

    def suggest(self) -> dict:
        self.trial_id += 1
        return self.model.suggest(self.trial_id).config

    def fit(self, df_features: pd.DataFrame, y: np.array):
        self.model = FIFOScheduler(
            config_space=self.config_space,
            metric=self.metric,
            mode=self.mode,
            points_to_evaluate=[],
            searcher="bayesopt",
        )
        for i, (xi, yi) in enumerate(zip(df_features.iterrows(), y)):
            self.model.on_trial_result(
                trial=Trial(trial_id=i, config=dict(xi[1]), creation_time=0),
                result={self.metric: yi},
            )
