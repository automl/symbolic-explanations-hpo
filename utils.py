import os
import time
import sympy
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from symbolic_meta_model_wrapper import (
    SymbolicMetaModelWrapper,
    SymbolicPursuitModelWrapper,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import configparser as cfgparse
from ConfigSpace import UniformIntegerHyperparameter

plt.style.use("tableau-colorblind10")


def get_output_dirs() -> [str, str, str]:
    """
    Create directory for current run as well as a plot and result subdirectory.
    """
    if not os.path.exists("runs"):
        os.makedirs("runs")
    run_dir = f"runs/run_{time.strftime('%Y%m%d-%H%M%S')}"
    res_dir = f"{run_dir}/results"
    plot_dir = f"{run_dir}/plots"
    os.makedirs(run_dir)
    os.makedirs(res_dir)
    os.makedirs(plot_dir)
    return run_dir, res_dir, plot_dir


def sort(x: np.ndarray, y: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Sort arrays according to values in x and reshape. Required for plotting.
    """
    idx = np.argsort(x, axis=0)
    x = x[idx].reshape(-1, 1)
    y = y[idx].reshape(-1)
    return x, y


def convert_symb(symb, n_dim: int = None, n_decimals: int = None) -> sympy.core.expr:
    """
    Convert a fitted symbolic regression to a simplified and potentially rounded mathematical expression.
    Warning: eval is used in this function, thus it should not be used on unsanitized input (see
    https://docs.sympy.org/latest/modules/core.html?highlight=eval#module-sympy.core.sympify).

    Parameters
    ----------
    symb: Fitted symbolic regressor to find a simplified expression for.
    n_dim: Number of input dimensions. If input has only a single dimension, X0 in expression is exchanged by x.
    n_decimals: If set, round floats in the expression to this number of decimals.

    Returns
    -------
    symb_conv: Converted mathematical expression.
    """
    if isinstance(symb, SymbolicRegressor):
        symb_str = str(symb._program)
    elif isinstance(symb, SymbolicMetaModelWrapper) or isinstance(
        symb, SymbolicPursuitModelWrapper
    ):
        symb_str = str(symb.expression())
    else:
        raise Exception("Unknown symbolic model")

    converter = {
        "sub": lambda x, y: x - y,
        "div": lambda x, y: x / y,
        "mul": lambda x, y: x * y,
        "add": lambda x, y: x + y,
        "neg": lambda x: -x,
        "pow": lambda x, y: x**y,
    }

    if len(symb_str) > 500:
        print(f"Expression of length {len(symb_str)} too long to convert, return raw string.")
        return symb_str

    symb_conv = sympy.simplify(
        sympy.sympify(symb_str.replace("[", "").replace("]", ""), locals=converter)
    )
    if n_dim == 1:
        x, X0 = sympy.symbols("x X0")
        symb_conv = symb_conv.subs(X0, x)
    if isinstance(symb, SymbolicPursuitModelWrapper):
        proj = symb.metamodel.get_projections()
        for i, p in enumerate(proj):
            for a in sympy.preorder_traversal(symb_conv):
                if isinstance(a, sympy.core.Symbol) and str(a) == f"P{i+1}":
                    symb_conv = symb_conv.subs(a, p)
    if n_decimals:
        # Make sure also floats deeper in the expression tree are rounded
        for a in sympy.preorder_traversal(symb_conv):
            if isinstance(a, sympy.core.numbers.Float):
                symb_conv = symb_conv.subs(a, round(a, n_decimals))

    return symb_conv


def append_scores(
    df_scores,
    col_name,
    symb_smac,
    symb_rand,
    X_train_smac,
    y_train_smac,
    X_train_rand,
    y_train_rand,
    X_test,
    y_test,
):
    """
    Append scores to Scores Dataframe.
    """
    df_scores[col_name] = {
        "mae_train_smac": mean_absolute_error(
            y_train_smac, symb_smac.predict(X_train_smac)
        ),
        "mae_train_rand": mean_absolute_error(
            y_train_rand, symb_rand.predict(X_train_rand)
        ),
        "mae_test_smac": mean_absolute_error(y_test, symb_smac.predict(X_test)),
        "mae_test_rand": mean_absolute_error(y_test, symb_rand.predict(X_test)),
        "mse_train_smac": mean_squared_error(
            y_train_smac, symb_smac.predict(X_train_smac)
        ),
        "mse_train_rand": mean_squared_error(
            y_train_rand, symb_rand.predict(X_train_rand)
        ),
        "mse_test_smac": mean_squared_error(y_test, symb_smac.predict(X_test)),
        "mse_test_rand": mean_squared_error(y_test, symb_rand.predict(X_test)),
        "r2_train_smac": r2_score(y_train_smac, symb_smac.predict(X_train_smac)),
        "r2_train_rand": r2_score(y_train_rand, symb_rand.predict(X_train_rand)),
        "r2_test_smac": r2_score(y_test, symb_smac.predict(X_test)),
        "r2_test_rand": r2_score(y_test, symb_rand.predict(X_test)),
    }
    return df_scores


def write_dict_to_cfg_file(dictionary: dict, target_file_path: str):
    parser = cfgparse.ConfigParser()
    section = "symbolic_regression"
    parser.add_section(section)

    for key in dictionary.keys():
        parser.set(section, key, str(dictionary[key]))
    with open(target_file_path, "w") as f:
        parser.write(f)


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
    X_train_smac,
    X_train_rand,
    X_test,
    y_test,
    symbolic_models,
    parameters,
    function_name,
    metric_name=None,
    function_expression=None,
    plot_dir=None,
):
    """
    In the 2D setting, create a plot showing the training points from SMAC and random sampling, as well as the true
    function and the functions fitted by symbolic models evaluated on a 2D grid.
    """

    LABEL_SIZE = 8
    TITLE_SIZE = 9
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

    fig, axes = plt.subplots(
        ncols=1, nrows=len(symbolic_models) + 1, constrained_layout=True
    )
    im = axes[0].pcolormesh(X_test[0], X_test[1], y_test, cmap="summer", shading="auto")
    if function_expression:
        axes[0].set_title(f"{function_expression}", fontsize=TITLE_SIZE)
    else:
        axes[0].set_title(f"{function_name}", fontsize=TITLE_SIZE)
    axes[0].set_xlabel(X0_name, fontsize=LABEL_SIZE)
    axes[0].set_ylabel(X1_name, fontsize=LABEL_SIZE)
    axes[0].set_xticks(dim_x)
    axes[0].set_yticks(dim_y)
    axes[0].set_xlim(X0_lower, X0_upper)
    axes[0].set_ylim(X1_lower, X1_upper)
    axes[0].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    axes[0].grid(alpha=0)

    for i, model_name in enumerate(symbolic_models):
        symbolic_model = symbolic_models[model_name]
        conv = convert_symb(symbolic_model, n_decimals=3)
        if len(str(conv)) < 70:
            label = f"{model_name}: {conv}"
        else:
            label = f"{model_name}:"
        im = axes[i + 1].pcolormesh(
            X_test[0],
            X_test[1],
            symbolic_model.predict(
                X_test.T.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0])
            )
            .reshape(X_test.shape[2], X_test.shape[1])
            .T,
            cmap="summer",
            shading="auto",
        )
        axes[i + 1].set_title(f"{label}", fontsize=TITLE_SIZE)
        axes[i + 1].set_xlabel(X0_name, fontsize=LABEL_SIZE)
        axes[i + 1].set_ylabel(X1_name, fontsize=LABEL_SIZE)
        axes[i + 1].set_xticks(dim_x)
        axes[i + 1].set_yticks(dim_y)
        axes[i + 1].set_xlim(X0_lower, X0_upper)
        axes[i + 1].set_ylim(X1_lower, X1_upper)
        axes[i + 1].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
        axes[i + 1].grid(alpha=0)
        X_train = None
        if "smac" in model_name:
            X_train = X_train_smac
        elif "rand" in model_name:
            X_train = X_train_rand
        if X_train is not None:
            axes[i + 1].scatter(
                X_train[0],
                X_train[1],
                color="midnightblue",
                zorder=2,
                marker=".",
                s=40,
                label="Train points",
            )

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
