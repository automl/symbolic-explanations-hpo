import os
import time
import sympy
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from symbolic_metamodeling.symbolic_meta_model_wrapper import SymbolicMetaModelWrapper, SymbolicMetaExpressionWrapper
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import configparser as cfgparse

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
    elif isinstance(symb, SymbolicMetaModelWrapper):
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

    symb_conv = sympy.simplify(sympy.sympify(symb_str, locals=converter))
    if n_dim == 1:
        x, X0 = sympy.symbols("x X0")
        symb_conv = symb_conv.subs(X0, x)
    if n_decimals:
        # Make sure also floats deeper in the expression tree are rounded
        for a in sympy.preorder_traversal(symb_conv):
            if isinstance(a, sympy.core.numbers.Float):
                symb_conv = symb_conv.subs(a, round(a, n_decimals))

    return symb_conv


def append_scores(
    df_scores,
    function,
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
    df_scores[function.expression] = {
        "mae_train_smac": mean_absolute_error(y_train_smac, symb_smac.predict(X_train_smac)),
        "mae_train_rand": mean_absolute_error(y_train_rand, symb_rand.predict(X_train_rand)),
        "mae_test_smac": mean_absolute_error(y_test, symb_smac.predict(X_test)),
        "mae_test_rand": mean_absolute_error(y_test, symb_rand.predict(X_test)),
        "mse_train_smac": mean_squared_error(y_train_smac, symb_smac.predict(X_train_smac)),
        "mse_train_rand": mean_squared_error(y_train_rand, symb_rand.predict(X_train_rand)),
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
    section = 'symbolic_regression'
    parser.add_section(section)

    for key in dictionary.keys():
        parser.set(section, key, str(dictionary[key]))
    with open(target_file_path, 'w') as f:
        parser.write(f)


def plot_symb1d(
    X_train_smac,
    y_train_smac,
    X_train_rand,
    y_train_rand,
    X_test,
    y_test,
    symbolic_models,
    function,
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
    plt.title(f"{function.expression}")
    plt.plot(
        X_test,
        y_test,
        color="C0",
        linewidth=3,
        label=f"True: {function.expression}",
    )

    colors = ["C1", "C8", "C6", "C9"]
    for i, model_name in enumerate(symbolic_models):
        symbolic_model = symbolic_models[model_name]

        conv = convert_symb(symbolic_model, n_dim=1, n_decimals=1)

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
            label=label
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
    epsilon = (X_test.max() - X_test.min()) / 100
    plt.xlim(X_test.min() - epsilon, X_test.max() + epsilon)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    leg = plt.legend(framealpha=0.)
    leg.get_frame().set_linewidth(0.0)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f"{plot_dir}/{function.name.lower().replace(' ', '_')}", dpi=800)
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
    function,
    plot_dir=None,
):
    """
    In the 2D setting, create a plot showing the training points from SMAC and random sampling, as well as the true
    function and the functions fitted by symbolic models evaluated on a 2D grid.
    """

    LABEL_SIZE = 8
    TITLE_SIZE = 9

    fig, axes = plt.subplots(ncols=len(symbolic_models) + 1, nrows=1)

    ax = plt.subplot(len(symbolic_models) + 1, 1, 1)
    plt.title(f"True: {function.expression}", fontsize=TITLE_SIZE)
    plt.pcolormesh(X_test[0], X_test[1], y_test, cmap='Greens', shading='auto')
    plt.xlabel("X0", fontsize=LABEL_SIZE)
    plt.ylabel("X1", fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=LABEL_SIZE)
    plt.grid(alpha=0)

    for i, model_name in enumerate(symbolic_models):
        symbolic_model = symbolic_models[model_name]

        conv = convert_symb(symbolic_model, n_decimals=3)

        if len(str(conv)) < 70:
            label = f"{model_name}: {conv}"
        else:
            label = f"{model_name}:"
        ax = plt.subplot(len(symbolic_models) + 1, 1, i + 2)
        plt.title(f"{label}", fontsize=TITLE_SIZE)
        plt.pcolormesh(X_test[0], X_test[1], symbolic_model.predict(X_test.T.reshape(100, 2)).reshape(10,10).T, 
                       cmap='Greens', shading='auto')
        plt.xlabel("X0", fontsize=LABEL_SIZE)
        plt.ylabel("X1", fontsize=LABEL_SIZE)
        plt.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=LABEL_SIZE)
        plt.grid(alpha=0)

        X_train = None
        if "smac" in model_name:
            X_train = X_train_smac
        elif "rand" in model_name:
            X_train = X_train_rand
        if X_train is not None:
            plt.scatter(
                X_train[0],
                X_train[1],
                color="blue",
                zorder=2,
                marker=".",
                s=5,
                label="Train points",
            )

    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower right', fontsize=LABEL_SIZE, framealpha=0.)
    leg.get_frame().set_linewidth(0.0)

    plt.tight_layout()
    if plot_dir:
        plt.savefig(f"{plot_dir}/{function.name.lower().replace(' ', '_')}", dpi=800)
    else:
        plt.show()
    plt.close()

    return fig