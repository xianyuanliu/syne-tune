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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42

from scipy.stats import norm

from benchmarking.nursery.benchmark_conformal.util import (
    figure_folder,
)
from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)

gp_name = "GP"
gp_color = "tab:green"
rs_name = "RS"
rs_color = "tab:blue"
qr_name = "QR"
qr_color = "orange"
cqr_name = "CQR"
cqr_color = "red"
kappa = 0.3
ylim = 2.5

fast_setup = False

if fast_setup:
    num_quantiles = 9
    num_fit_examples = 1000
    num_samples_model = 10000
    num_test_samples = 200
else:
    num_quantiles = 49
    num_fit_examples = 1000
    num_samples_model = 10000
    num_test_samples = 200


def draw_Xy_sine(config_space, num_samples: int):
    X = pd.DataFrame(
        [{k: v.sample() for k, v in config_space.items()} for _ in range(num_samples)]
    )
    y = np.random.normal(
        loc=np.zeros_like(X.values.flatten()),
        scale=np.sin(X.values.flatten()) ** 2 + kappa,
    )
    return X, y


np.random.seed(42)

config_space = {"x": uniform(0, 2 * np.pi)}

X_train, y_train = draw_Xy_sine(num_samples=num_fit_examples, config_space=config_space)
X_test = pd.DataFrame(np.linspace(0, 2 * np.pi, num_test_samples), columns=["x"])
y_test = np.random.normal(loc=np.zeros_like(X_test), scale=np.sin(X_test) ** 2 + kappa)

gp_samples = (
    X_test.sample(n=num_samples_model, replace=True).values.reshape(-1).tolist()
)
qr_model = QuantileRegressionSurrogateModel(
    config_space=config_space,
    mode="max",
    quantiles=num_quantiles,
    verbose=True,
)
qr_model.fit(df_features=X_train, y=y_train, ncandidates=X_test)
qr_samples = [
    qr_model.suggest(replace_config=False)["x"] for _ in range(num_samples_model)
]


def plot_thompson(ax):
    np.random.seed(10)

    xs = np.linspace(0, 2 * np.pi)
    stds = np.sin(xs) ** 2 + kappa
    alphas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    y_quantiles = np.array(
        [norm.ppf(q=alpha, loc=0 * stds, scale=stds) for alpha in alphas]
    )
    cmap = cm.get_cmap("viridis")
    for q, yq in zip(alphas, y_quantiles):
        label = "$q_{" + str(q) + "}(x)$"
        ax.plot(xs, yq, label=label, color=cmap(q))
    n_sample = 10
    index_candidates = [int(len(xs) / n_sample * i) - 2 for i in range(1, n_sample)]
    x_candidates = [xs[j] for j in index_candidates]
    choices = [2, 6, 0, 5, 6, 2, 5, 2, 0]
    y_candidates = [y_quantiles[choices[i], j] for i, j in enumerate(index_candidates)]
    best_index = np.argmin(y_candidates)
    best_x = x_candidates[best_index]
    best_y = y_candidates[best_index]

    for i in range(len(index_candidates)):
        x, y = x_candidates[i], y_candidates[i]
        ax.axvline(x=x_candidates[i], color="black", alpha=0.8, lw=0.2, linestyle="--")
        ax.plot(
            [x],
            [y],
            marker="o",
            markersize=10,
            linestyle="",
            markeredgecolor="black",
            markerfacecolor="black",
            label="Samples $\\tilde{y}(x)$" if i == 0 else None,
        )

        ax.plot(
            [best_x],
            [best_y],
            marker="o",
            markersize=10,
            linestyle="",
            markeredgecolor="black",
            markerfacecolor="red",
            label="Samples $\\tilde{y}(x)$" if i == 0 else None,
        )
        ax.axvline(x=best_x, color="r", alpha=0.9, lw=2.0, linestyle="--")

        ax.set_xlim(xs[0], xs[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            x=best_x + 0.2,
            y=best_y,
            s="$x^* = argmin_{x\in X} \\tilde{y}(x)$",
            color="red",
            size=14,
        )
    ax.set_title("Illustration of the proposed sampling")


def plot_samples(ax, x, X, y):
    ax.plot(
        X,
        y,
        label=r"$f(x)~\sim~\mathcal{N}(0, \rho(x)^2)$",
        marker="+",
        linestyle="None",
    )
    ax.plot(x, np.sin(x) ** 2 + kappa, label=r"$\pm \rho(x)$", color="black")
    ax.plot(x, -(np.sin(x) ** 2 + kappa), color="black")
    ax.set_ylim(-ylim, ylim)
    vline_kwargs = dict(color="red", ymin=-3, ymax=3, alpha=0.5, linestyles="--")
    ax.vlines(np.pi / 2, **vline_kwargs)
    ax.vlines(3 * np.pi / 2, **vline_kwargs)
    ax.set_xlim(left=0, right=2 * np.pi)
    ax.legend(loc="upper right")
    ax.set_title(r"Samples from $f(x)~\sim~\mathcal{N}(0, \rho(x)^2)$")


def plot_acq_function(ax, gp_samples, qr_samples):
    sns.histplot(
        gp_samples,
        label=gp_name,
        ax=ax,
        bins=32,
        stat="probability",
        color=gp_color,
    )

    sns.histplot(
        qr_samples,
        label=qr_name,
        ax=ax,
        stat="probability",
        bins=32,
        color=qr_color,
    )
    ax.legend(loc="upper right")
    ax.set_title("Acquisition function")


X = X_train
y = y_train

f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, figsize=(13, 4))
x = np.linspace(0, 2 * np.pi)
plot_samples(ax0, x, X, y)
plot_acq_function(ax2, gp_samples, qr_samples)

for ax in [ax0, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(None)
plot_thompson(ax1)
ax1.set_ylim(-ylim, ylim)
plt.tight_layout()
figure_path = figure_folder(subpath="./") / f"method-illustration.pdf"
plt.savefig(figure_path)
plt.show()
