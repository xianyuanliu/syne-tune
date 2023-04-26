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
import numpy as np
from scipy.stats import norm


def quantile_coverage(
    y_true: np.array,
    quantiles_pred: np.array,
):
    """
    :param y_true: shape (n_samples)
    :param quantiles_pred: shape (n_samples, num_quantiles)
    :return:
    """
    num_samples, num_quantiles = quantiles_pred.shape

    quantiles_coverage = np.zeros(num_quantiles)
    for i, _ in enumerate(quantiles_coverage):
        quantiles_coverage[i] = np.mean(1.0 * (y_true < quantiles_pred[:, i]))

    return quantiles_coverage


def rmsce_calibration(
    y_true: np.array,
    quantiles_pred: np.array,
    quantiles: np.array,
):
    """
    Return root mean squared calibration error which consists in $(p_j - \tilde{p_j})^2$ where $p_j$ is a given quantile
    in [0, 1] and $\tilde{p_j}$ is the coverage of this quantile for the predictions, eg the percent of time this
    quantile was above the target.
    https://arxiv.org/pdf/1807.00263.pdf
    :param y_true: true value with shape (num_samples,)
    :param quantiles_pred: tensor with shape (num_samples, num_quantiles) representing the quantiles predicted
    :param quantiles: the `num_quantiles` considered
    :return:
    """
    assert len(y_true) == len(quantiles_pred)
    assert quantiles_pred.shape[1] == len(quantiles)

    coverage = quantile_coverage(y_true=y_true, quantiles_pred=quantiles_pred)

    # return np.sqrt(((coverage - quantiles) ** 2).mean())
    return (np.abs(coverage - quantiles)).mean()


def rmse_calibration_from_normal(
    y_true: np.array,
    mu_pred: np.array,
    std_pred: np.array,
    quantiles: np.array,
):
    inv_cdf = norm.ppf(quantiles).reshape(1, -1)
    return rmsce_calibration(
        y_true=y_true,
        quantiles_pred=inv_cdf * np.expand_dims(std_pred, 1)
        + np.expand_dims(mu_pred, 1),
        quantiles=quantiles,
    )


if __name__ == "__main__":
    num_samples = 1000
    num_quantiles = 9
    quantiles = np.linspace(0, 1, num=num_quantiles + 2)[1:-1]

    y_true = np.random.normal(0, 3, size=num_samples)
    quantiles_pred = np.array([norm.ppf(quantiles) for _ in range(num_samples)])
    print("calibration error of wrong distribution")
    print(quantile_coverage(y_true=y_true, quantiles_pred=quantiles_pred))
    print(
        rmsce_calibration(
            y_true=y_true, quantiles_pred=quantiles_pred, quantiles=quantiles
        )
    )

    print("calibration error of right distribution")
    print(quantile_coverage(y_true=y_true, quantiles_pred=quantiles_pred))
    print(
        rmsce_calibration(
            y_true=np.random.normal(0, 1, size=num_samples),
            quantiles_pred=quantiles_pred,
            quantiles=quantiles,
        )
    )

    mu, std = 1, 4
    y_true = np.random.normal(mu, std, size=num_samples)
    inv_cdf = norm.ppf(quantiles).reshape(1, -1)
    mu_pred = mu * np.ones_like(y_true)
    std_pred = np.ones_like(y_true) * std
    calib = rmsce_calibration(
        y_true=y_true,
        quantiles_pred=inv_cdf * np.expand_dims(std_pred, 1)
        + np.expand_dims(mu_pred, 1),
        quantiles=quantiles,
    )
    print(calib)

    mu, std = 10, 20
    y_true = np.random.normal(mu, std, size=num_samples)
    inv_cdf = norm.ppf(quantiles).reshape(1, -1)
    mu_pred = mu * np.ones_like(y_true)
    std_pred = np.ones_like(y_true) * std
    calib = rmsce_calibration(
        y_true=y_true,
        quantiles_pred=inv_cdf * np.expand_dims(std_pred, 1)
        + np.expand_dims(mu_pred, 1),
        quantiles=quantiles,
    )

    print(calib)

    calib = rmse_calibration_from_normal(
        y_true=y_true,
        mu_pred=mu_pred,
        std_pred=std_pred,
        quantiles=quantiles,
    )

    print(calib)
