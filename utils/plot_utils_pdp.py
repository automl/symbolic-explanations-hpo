import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import UniformIntegerHyperparameter


def plot_symb2d_subplots(
    X_train_list,
    X_test,
    y_test,
    predictions_test,
    parameters,
    function_name,
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
    y_values = np.concatenate([pred.reshape(-1, 1) for pred in pred_test])
    y_values = np.concatenate((y_test.reshape(-1, 1), y_values))
    vmin, vmax = min(y_values), max(y_values)

    if parameters[0].log:
        X0_test = np.log(X_test[0])
    else:
        X0_test = X_test[0]
    if parameters[1].log:
        X1_test = np.log(X_test[1])
    else:
        X1_test = X_test[1]

    fig = plt.figure(figsize=(15, 6))

    ax = plt.subplot(2, 2, 1)
    im = ax.pcolormesh(
        X0_test,
        X1_test,
        y_test,
        cmap="summer",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"GP Baseline (Partial Dependence)", fontsize=TITLE_SIZE)
    ax.set_ylabel(X1_name, fontsize=TITLE_SIZE, labelpad=5)
    ax.set_xticks(dim_x)
    ax.set_yticks(dim_y)
    ax.set_xlim(X0_lower, X0_upper)
    ax.set_ylim(X1_lower, X1_upper)
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    ax.grid(alpha=0)

    for ind, model_name in enumerate(predictions_test):
        print(model_name)
        label = model_name
        i = 2
        ax = plt.subplot(1, 2, i)

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
        ax.set_xlabel(X0_name, fontsize=TITLE_SIZE)
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
