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
from argparse import ArgumentParser
from pathlib import Path

from coolname import generate_slug
from sagemaker.pytorch import PyTorch
from tqdm import tqdm

from benchmarking.nursery.benchmark_conformal.baselines import (
    methods,
    Methods,
)
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default=generate_slug(2)
    )
    parser.add_argument(
        "--experiment_subtag", type=str, required=False, default=random_string(4)
    )
    parser.add_argument("--n_workers", type=int, required=False, default=4)
    parser.add_argument("--num_seeds", type=int, required=False, default=30)
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    experiment_subtag = args.experiment_subtag
    num_seeds = args.num_seeds
    print(args)
    print(f"Methods defined: {list(methods.keys())}")
    methods_selected = [
        Methods.RS,
        Methods.REA,
        Methods.TPE,
        Methods.GP,
        Methods.BORE,
        Methods.QR,
        Methods.CQR,
        # MF
        Methods.ASHA,
        Methods.BOHB,
        Methods.HYPERTUNE,
        Methods.MOBSTER,
        # # ours
        Methods.multifidelity(Methods.QR),
        Methods.multifidelity(Methods.CQR),
        Methods.multifidelity(Methods.BORE),
        Methods.multifidelity(Methods.REA),
        Methods.multifidelity(Methods.GP),
    ]
    # methods_selected = [Methods.MOBSTER, Methods.HYPERTUNE]
    print(f"{len(methods_selected)} methods selected: {methods_selected}")

    # methods for which we spawn one job per seed due to speed
    # 5 * num_seeds jobs, 150 ATM
    distributed_methods = [
        Methods.MOBSTER,
        Methods.HYPERTUNE,
        Methods.multifidelity(Methods.QR),
        Methods.multifidelity(Methods.CQR),
        Methods.multifidelity(Methods.BORE),
    ]
    for method in tqdm(methods_selected):
        assert method in methods
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            # instance_type="local",
            checkpoint_s3_uri=s3_experiment_path(
                tuner_name=method, experiment_name=experiment_tag
            ),
            instance_type="ml.m5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=3600 * 24 * 5,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
            hyperparameters={
                "experiment_tag": experiment_tag,
                "subtag": experiment_subtag,
                "method": method,
                "n_workers": args.n_workers,
            },
        )

        if method not in distributed_methods:
            print(f"{experiment_tag}-{method}")
            sm_args["hyperparameters"]["num_seeds"] = num_seeds
            est = PyTorch(**sm_args)
            est.fit(
                job_name=f"{experiment_tag}-{method}-{experiment_subtag}", wait=False
            )
        else:
            # For mobster, we schedule one job per seed as the method takes much longer
            for seed in range(num_seeds):
                print(f"{experiment_tag}-{method}-{seed}")
                sm_args["hyperparameters"].update(
                    {
                        "num_seeds": seed,
                        "run_all_seed": 0,
                    }
                )
                est = PyTorch(**sm_args)
                est.fit(
                    job_name=f"{experiment_tag}-{method}-{seed}-{experiment_subtag}",
                    wait=False,
                )
