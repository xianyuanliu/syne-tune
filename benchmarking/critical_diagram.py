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
import pandas as pd


def draw_cd_diagram(
    df: pd.DataFrame,
    method_column: str,
    rank_column: str,
    dataset_column: str,
    alpha: float = 0.05,
    title: str = None,
    folder: str = None,
):
    """
    :param df: dataframe that contains all methods results
    :param method_column: column that contains the method name
    :param rank_column: column that contains the rank, values should be in [0, 1] where lower is better.
    :param dataset_column: column that contains dataset
    :param alpha: statistical significancy for statistical tests
    :param title: title of the figure
    :param folder: folder to write the figure to
    :return:
    """
    pass


if __name__ == "__main__":
    df_perf = pd.read_csv(
        "/Users/dsalina/Documents/code/cd-diagram/example.csv", index_col=False
    )

    draw_cd_diagram(
        df=df_perf,
        method_column="classifier_name",
        dataset_column="dataset_name",
        rank_column="accuracy",
        title="Accuracy",
    )
