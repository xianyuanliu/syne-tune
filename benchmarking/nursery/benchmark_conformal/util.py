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
from pathlib import Path


def figure_folder(subpath=None):
    # path used to store figures in the experiments, subpath can be asked on this folder, the folder returned is created
    # avoid storing in benchmarking in case the folder is sent to S3 for experiments.
    res = Path(__file__).parent.parent.parent.parent / "figures" / "icml2023"
    if subpath:
        res = res / subpath
    res.mkdir(parents=True, exist_ok=True)
    return res
