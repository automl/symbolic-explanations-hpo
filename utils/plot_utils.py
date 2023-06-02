import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import UniformIntegerHyperparameter

from utils.run_utils import convert_symb


def sort(x: np.ndarray, y: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Sort arrays according to values in x and reshape. Required for plotting.
    """
    idx = np.argsort(x, axis=0)
    x = x[idx].reshape(-1, 1)
    y = y[idx].reshape(-1)
    return x, y

def plot_symb1d(
    X_train_smac,
    y_train_smac,
    X_train_rand,
    y_train_rand,
    X_test,
    y_test,
    xlabel,
    ylabel,
    symbolic_models,
    function_name,
    xmin=None,
    xmax=None,
    function_expression=None,
    plot_dir=None,
):
    """
    In the 1D setting, create a plot showing the training points from SMAC and random sampling, as well as the true
    function and the functions fitted by symbolic models.
    """
    X_train_smac, y_train_smac = sort(X_train_smac, y_train_smac)
    X_train_rand, y_train_rand = sort(X_train_rand, y_train_rand)
    X_test, y_test = sort(X_test, y_test)
    fig = plt.figure(figsize=(8, 5))
    if function_expression:
        plt.title(f"{function_expression}")
    else:
        plt.title(f"{function_name}")
    if function_expression:
        plt.plot(
            X_test,
            y_test,
            color="C0",
            linewidth=3,
            label=f"True: {function_expression}",
        )
    else:
        plt.scatter(
            X_test,
            y_test,
            color="C0",
            zorder=3,
            marker="+",
            s=20,
            label=f"Test",
        )
    colors = ["C1", "C8", "C6", "C9"]
    for i, model_name in enumerate(symbolic_models):
        symbolic_model = symbolic_models[model_name]

        conv = convert_symb(symbolic_model, n_dim=1, n_decimals=3)

        if len(str(conv)) < 70:
            label = f"{model_name}: {conv}"
        else:
            conv = convert_symb(symbolic_model, n_dim=1, n_decimals=1)
            if len(str(conv)) < 70:
                label = f"{model_name}: {conv}"
            else:
                label = f"{model_name}:"
        plt.plot(
            X_test,
            symbolic_model.predict(X_test),
            color=colors[i],
            linewidth=2,
            label=label,
        )
    plt.scatter(
        X_train_smac,
        y_train_smac,
        color="C5",
        zorder=2,
        marker="^",
        s=40,
        label=f"Train points (smac)",
    )
    plt.scatter(
        X_train_rand,
        y_train_rand,
        color="C3",
        zorder=2,
        marker="P",
        s=40,
        label="Train points (random)",
    )
    xmax = xmax if xmax else X_test.max()
    xmin = xmin if xmin else X_test.min()
    epsilon = (xmax - xmin) / 100
    plt.xlim(xmin - epsilon, xmax + epsilon)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    leg = plt.legend(framealpha=0.0)
    leg.get_frame().set_linewidth(0.0)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f"{plot_dir}/{function_name}", dpi=800)
    else:
        plt.show()
    plt.close()
    return fig


def plot_symb2d(
    X_train_list,
    X_test,
    y_test,
    predictions_test,
    parameters,
    function_name,
    use_same_scale=True,
    metric_name=None,
    function_expression=None,
    plot_dir=None,
    filename=None,
):
    """
    In the 2D setting, create a plot showing the training points from SMAC and another sampling, as well as the true
    function and the functions fitted by symbolic models evaluated on a 2D grid.
    """

    LABEL_SIZE = 10
    TITLE_SIZE = 11
    X0_name = (
        "X0"
        if parameters[0].name == "X0"
        else f"log({parameters[0].name})"
        if parameters[0].log
        else f"{parameters[0].name}"
    )
    X1_name = (
        "X1"
        if parameters[1].name == "X1"
        else f"log({parameters[1].name})"
        if parameters[1].log
        else f"{parameters[1].name}"
    )
    if parameters[0].log:
        X0_upper = np.log(parameters[0].upper)
        X0_lower = np.log(parameters[0].lower)
    else:
        X0_upper = parameters[0].upper
        X0_lower = parameters[0].lower
    if isinstance(parameters[0], UniformIntegerHyperparameter):
        dim_x = X_test[0][0].astype(int)
        if parameters[0].log:
            dim_x = np.log(dim_x)
    else:
        step_x = (X0_upper - X0_lower) / X_test.shape[2]
        if parameters[0].log:
            dim_x = np.arange(np.log(np.min(X_test[0])), np.log(np.max(X_test[0])) + step_x / 2, step_x)
        else:
            dim_x = np.arange(np.min(X_test[0]), np.max(X_test[0]) + step_x / 2, step_x)
    if parameters[1].log:
        X1_upper = np.log(parameters[1].upper)
        X1_lower = np.log(parameters[1].lower)
        step_y = 1 / 2 * (np.log(np.max(X_test[1])) - np.log(np.min(X_test[1])))
        dim_y = np.arange(np.log(np.min(X_test[1])), np.log(np.max(X_test[1])) + step_y / 2, step_y)
    else:
        X1_upper = parameters[1].upper
        X1_lower = parameters[1].lower
        step_y = 1 / 2 * (np.max(X_test[1]) - np.min(X_test[1]))
        dim_y = np.arange(np.min(X_test[1]), np.max(X_test[1]) + step_y / 2, step_y)


    fig, axes = plt.subplots(
        ncols=1, nrows=len(predictions_test) + 1, constrained_layout=True, figsize=(8, 5)
    )

    pred_test = []
    for i, model_name in enumerate(predictions_test):
        pred_test.append(predictions_test[model_name])
    y_values = np.concatenate(
        (
            y_test.reshape(-1, 1),
            pred_test[0].reshape(-1, 1),
            pred_test[1].reshape(-1, 1),
        )
    )
    if use_same_scale:
        vmin, vmax = min(y_values), max(y_values)
    else:
        vmin, vmax = None, None

    if parameters[0].log:
        X0_test = np.log(X_test[0])
    else:
        X0_test = X_test[0]
    if parameters[1].log:
        X1_test = np.log(X_test[1])
    else:
        X1_test = X_test[1]

    im = axes[0].pcolormesh(
        X0_test,
        X1_test,
        y_test,
        cmap="summer",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    if function_expression:
        axes[0].set_title(f"True: {function_expression}", fontsize=TITLE_SIZE)
    else:
        axes[0].set_title(f"Ground Truth", fontsize=TITLE_SIZE)
    #axes[0].set_xlabel(X0_name, fontsize=TITLE_SIZE)
    #axes[0].set_ylabel(X1_name, fontsize=TITLE_SIZE)
    axes[0].set_xticks(dim_x)
    axes[0].set_yticks(dim_y)
    axes[0].set_xlim(X0_lower, X0_upper)
    axes[0].set_ylim(X1_lower, X1_upper)
    axes[0].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    axes[0].grid(alpha=0)

    if not use_same_scale:
        cbar = fig.colorbar(im, ax=axes[0])
        cbar.set_label(r'True $\mathcal{L}$', fontsize=TITLE_SIZE, rotation=270, labelpad=10)
        cbar.ax.tick_params(labelsize=LABEL_SIZE)

    for i, model_name in enumerate(predictions_test):
        label = model_name
        im = axes[i + 1].pcolormesh(
            X0_test,
            X1_test,
            pred_test[i],
            cmap="summer",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axes[i + 1].set_title(f"Prediction: {label}", fontsize=TITLE_SIZE)
        if i == len(predictions_test) - 1:
            axes[i + 1].set_xlabel(X0_name, fontsize=TITLE_SIZE)
        axes[i + 1].set_ylabel(X1_name, fontsize=TITLE_SIZE)
        axes[i + 1].set_xticks(dim_x)
        axes[i + 1].set_yticks(dim_y)
        axes[i + 1].set_xlim(X0_lower, X0_upper)
        axes[i + 1].set_ylim(X1_lower, X1_upper)
        axes[i + 1].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
        axes[i + 1].grid(alpha=0)
        X_train = X_train_list[i]
        if X_train is not None:
            if parameters[0].log:
                X_train[0] = np.log(X_train[0])
            if parameters[1].log:
                X_train[1] = np.log(X_train[1])
            axes[i + 1].scatter(
                X_train[0],
                X_train[1],
                color="midnightblue",
                zorder=2,
                marker=".",
                s=40,
                label="SR Train Points",
            )
    if not use_same_scale:
        cbar = fig.colorbar(im, ax=axes[1:], shrink=0.4)
        if metric_name:
            cbar.set_label(metric_name, fontsize=TITLE_SIZE, rotation=270, labelpad=15)
        else:
            cbar.set_label("f(X0, X1)", fontsize=TITLE_SIZE, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=LABEL_SIZE)
    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower right", bbox_to_anchor=(1., -0.022), fontsize=TITLE_SIZE, framealpha=0.0
    )
    leg.get_frame().set_linewidth(0.0)
    if use_same_scale:
        cbar = fig.colorbar(im, ax=axes, shrink=0.4)
        if metric_name:
            cbar.set_label(metric_name, fontsize=TITLE_SIZE, rotation=270, labelpad=15)
        else:
            cbar.set_label("f(X0, X1)", fontsize=TITLE_SIZE, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=LABEL_SIZE)
    if plot_dir:
        if filename:
            plt.savefig(f"{plot_dir}/{filename}", dpi=800)
        else:
            plt.savefig(f"{plot_dir}/{function_name}", dpi=800)
    else:
        plt.show()
    plt.close()

    return fig

def plot_symb2d_subplots(
    X_train_list,
    X_test,
    y_test,
    predictions_test,
    parameters,
    function_name,
    use_same_scale=True,
    metric_name=None,
    plot_dir=None,
    filename=None,
):
    """
    In the 2D setting, create a plot showing the training points from SMAC and another sampling, as well as the true
    function and the functions fitted by symbolic models evaluated on a 2D grid.
    """

    LABEL_SIZE = 12
    TITLE_SIZE = 15
    X0_name = (
        "X0"
        if parameters[0].name == "X0"
        else f"log({parameters[0].name})"
        if parameters[0].log
        else f"{parameters[0].name}"
    )
    X1_name = (
        "X1"
        if parameters[1].name == "X1"
        else f"log({parameters[1].name})"
        if parameters[1].log
        else f"{parameters[1].name}"
    )
    if parameters[0].log:
        X0_upper = np.log(parameters[0].upper)
        X0_lower = np.log(parameters[0].lower)
    else:
        X0_upper = parameters[0].upper
        X0_lower = parameters[0].lower
    if isinstance(parameters[0], UniformIntegerHyperparameter):
        dim_x = X_test[0][0].astype(int)
        if parameters[0].log:
            dim_x = np.log(dim_x)
    else:
        step_x = (X0_upper - X0_lower) / X_test.shape[2]
        if parameters[0].log:
            dim_x = np.arange(np.log(np.min(X_test[0])), np.log(np.max(X_test[0])) + step_x / 2, step_x)
        else:
            dim_x = np.arange(np.min(X_test[0]), np.max(X_test[0]) + step_x / 2, step_x)
    if parameters[1].log:
        X1_upper = np.log(parameters[1].upper)
        X1_lower = np.log(parameters[1].lower)
        step_y = 1 / 2 * (np.log(np.max(X_test[1])) - np.log(np.min(X_test[1])))
        dim_y = np.arange(np.log(np.min(X_test[1])), np.log(np.max(X_test[1])) + step_y / 2, step_y)
    else:
        X1_upper = parameters[1].upper
        X1_lower = parameters[1].lower
        step_y = 1 / 2 * (np.max(X_test[1]) - np.min(X_test[1]))
        dim_y = np.arange(np.min(X_test[1]), np.max(X_test[1]) + step_y / 2, step_y)

    pred_test = []
    for i, model_name in enumerate(predictions_test):
        pred_test.append(predictions_test[model_name])
    y_values = np.concatenate(
        (
            y_test.reshape(-1, 1),
            pred_test[0].reshape(-1, 1),
            pred_test[1].reshape(-1, 1),
            pred_test[2].reshape(-1, 1),
            pred_test[3].reshape(-1, 1),
        )
    )
    if use_same_scale:
        vmin, vmax = min(y_values), max(y_values)
    else:
        vmin, vmax = None, None

    if parameters[0].log:
        X0_test = np.log(X_test[0])
    else:
        X0_test = X_test[0]
    if parameters[1].log:
        X1_test = np.log(X_test[1])
    else:
        X1_test = X_test[1]

    fig = plt.figure(figsize=(15, 6))

    ax = plt.subplot(3, 2, 1)
    im = ax.pcolormesh(
        X0_test,
        X1_test,
        y_test,
        cmap="summer",
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Ground Truth", fontsize=TITLE_SIZE)
    ax.set_xticks(dim_x)
    ax.set_yticks(dim_y)
    ax.set_xlim(X0_lower, X0_upper)
    ax.set_ylim(X1_lower, X1_upper)
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    ax.grid(alpha=0)

    for ind, model_name in enumerate(predictions_test):
        label = model_name
        if model_name == "GP Baseline":
            i = 3
        elif model_name == "SR (BO)":
            i = 2
        elif model_name == "SR (Random)":
            i = 4
        elif model_name == "SR (BO-GP)":
            i = 6
        ax = plt.subplot(3, 2, i)

        im = ax.pcolormesh(
            X0_test,
            X1_test,
            pred_test[ind],
            cmap="summer",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Prediction: {label}", fontsize=TITLE_SIZE)
        if i == 3 or i == 6:
            ax.set_xlabel(X0_name, fontsize=TITLE_SIZE)
        if i == 3:
            ax.set_ylabel(X1_name, fontsize=TITLE_SIZE, labelpad=5)
        ax.set_xticks(dim_x)
        ax.set_yticks(dim_y)
        ax.set_xlim(X0_lower, X0_upper)
        ax.set_ylim(X1_lower, X1_upper)
        ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
        ax.grid(alpha=0)
        X_train = X_train_list[ind]
        if X_train is not None:
            if parameters[0].log:
                X_train[0] = np.log(X_train[0])
            if parameters[1].log:
                X_train[1] = np.log(X_train[1])
            ax.scatter(
                X_train[0],
                X_train[1],
                color="midnightblue",
                zorder=2,
                marker=".",
                s=70,
                label="SR Train Points",
            )
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower right", bbox_to_anchor=(0.35, 0.03), fontsize=TITLE_SIZE, framealpha=0.0, handletextpad=0.05
    )
    for handle in leg.legendHandles:
        handle.set_sizes([100])
    leg.get_frame().set_linewidth(0.0)
    cbar_ax = fig.add_axes([0.12, 0.24, 0.3, 0.04])
    cbar = fig.colorbar(im, ax=ax, cax=cbar_ax, shrink=0.4, orientation="horizontal")
    cbar.set_label(metric_name, fontsize=TITLE_SIZE, labelpad=6)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)
    plt.tight_layout()
    if plot_dir:
        if filename:
            plt.savefig(f"{plot_dir}/{filename}", dpi=800)
        else:
            plt.savefig(f"{plot_dir}/{function_name}", dpi=800)
    else:
        plt.show()
    plt.close()

    return fig


def plot_symb2d_surrogate(
    X_train_smac,
    X_test,
    y_test,
    y_test_surrogate,
    symbolic_models,
    parameters,
    function_name,
    metric_name=None,
    function_expression=None,
    plot_dir=None,
):
    """
    In the 2D setting, create a plot showing the training points from SMAC as well as the
    true function and the functions fitted by the surrogate model evaluated on a 2D grid.
    Furthermore, the function fitted by a symbolic model on the surrogate values of each
    grid point evaluated on the grid is shown.
    """

    LABEL_SIZE = 9
    X0_name = (
        "X0"
        if parameters[0].name == "X0"
        else f"X0: log({parameters[0].name})"
        if parameters[0].log
        else f"X0: {parameters[0].name}"
    )
    X1_name = (
        "X1"
        if parameters[1].name == "X1"
        else f"X1: log({parameters[1].name})"
        if parameters[1].log
        else f"X1: {parameters[1].name}"
    )
    if parameters[0].log:
        X0_upper = np.log(parameters[0].upper)
        X0_lower = np.log(parameters[0].lower)
    else:
        X0_upper = parameters[0].upper
        X0_lower = parameters[0].lower
    if isinstance(parameters[0], UniformIntegerHyperparameter):
        dim_x = X_test[0][0].astype(int)
    else:
        step_x = (X0_upper - X0_lower) / X_test.shape[2]
        dim_x = np.arange(np.min(X_test[0]), np.max(X_test[0]) + step_x / 2, step_x)
    if parameters[1].log:
        X1_upper = np.log(parameters[1].upper)
        X1_lower = np.log(parameters[1].lower)
    else:
        X1_upper = parameters[1].upper
        X1_lower = parameters[1].lower
    step_y = 1 / 2 * (np.max(X_test[1]) - np.min(X_test[1]))
    dim_y = np.arange(np.min(X_test[1]), np.max(X_test[1]) + step_y / 2, step_y)

    model_name = "Symb-smac"
    symbolic_model = symbolic_models[model_name]
    pred_test = symbolic_model.predict(
        X_test.T.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0])
    )
    y_values = np.concatenate(
        (
            y_test.reshape(-1, 1),
            y_test_surrogate.reshape(-1, 1),
            pred_test.reshape(-1, 1),
        )
    )
    vmin, vmax = min(y_values), max(y_values)

    fig, axes = plt.subplots(
        ncols=1, nrows=len(symbolic_models) + 1, constrained_layout=True, figsize=(8, 5)
    )
    im = axes[0].pcolormesh(X_test[0], X_test[1], y_test, cmap="summer", shading="auto")
    if function_expression:
        axes[0].set_title(f"True: {function_expression}", fontsize=LABEL_SIZE)
    else:
        axes[0].set_title(f"True", fontsize=LABEL_SIZE)
    axes[0].set_xlabel(X0_name, fontsize=LABEL_SIZE)
    axes[0].set_ylabel(X1_name, fontsize=LABEL_SIZE)
    axes[0].set_xticks(dim_x)
    axes[0].set_yticks(dim_y)
    axes[0].set_xlim(X0_lower, X0_upper)
    axes[0].set_ylim(X1_lower, X1_upper)
    axes[0].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    axes[0].grid(alpha=0)

    im = axes[1].pcolormesh(
        X_test[0],
        X_test[1],
        y_test_surrogate.reshape(X_test.shape[1], X_test.shape[2]),
        cmap="summer",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].scatter(
        X_train_smac[0],
        X_train_smac[1],
        color="midnightblue",
        zorder=2,
        marker=".",
        s=40,
        label="Train points",
    )
    axes[1].set_title(f"Surrogate Model", fontsize=LABEL_SIZE)
    axes[1].set_xlabel(X0_name, fontsize=LABEL_SIZE)
    axes[1].set_ylabel(X1_name, fontsize=LABEL_SIZE)
    axes[1].set_xticks(dim_x)
    axes[1].set_yticks(dim_y)
    axes[1].set_xlim(X0_lower, X0_upper)
    axes[1].set_ylim(X1_lower, X1_upper)
    axes[1].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    axes[1].grid(alpha=0)

    conv = convert_symb(symbolic_model, n_decimals=3)
    if len(str(conv)) < 80:
        label = f"{model_name}: {conv}"
    else:
        label = f"{model_name}:"
    im = axes[2].pcolormesh(
        X_test[0],
        X_test[1],
        pred_test.reshape(X_test.shape[1], X_test.shape[2]),
        cmap="summer",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].scatter(
        X_test[0],
        X_test[1],
        color="midnightblue",
        zorder=2,
        marker=".",
        s=40,
        label="Train points",
    )
    axes[2].set_title(f"{label}", fontsize=LABEL_SIZE)
    axes[2].set_xlabel(X0_name, fontsize=LABEL_SIZE)
    axes[2].set_ylabel(X1_name, fontsize=LABEL_SIZE)
    axes[2].set_xticks(dim_x)
    axes[2].set_yticks(dim_y)
    axes[2].set_xlim(X0_lower, X0_upper)
    axes[2].set_ylim(X1_lower, X1_upper)
    axes[2].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)

    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower right", fontsize=LABEL_SIZE, framealpha=0.0
    )
    leg.get_frame().set_linewidth(0.0)
    cbar = fig.colorbar(im, ax=axes, shrink=0.4)
    if metric_name:
        cbar.set_label(metric_name, fontsize=LABEL_SIZE, rotation=270, labelpad=10)
    else:
        cbar.set_label("f(X0, X1)", fontsize=LABEL_SIZE, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)
    if plot_dir:
        plt.savefig(f"{plot_dir}/{function_name}", dpi=800)
    else:
        plt.show()
    plt.close()

    return fig
