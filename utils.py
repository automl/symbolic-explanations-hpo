import os
import time
import sympy
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from symbolic_metamodeling.symbolic_meta_model_wrapper import SymbolicMetaModelWrapper
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


def convert_symb(symb, n_decimals: int = None) -> sympy.core.expr:
    """
    Convert a fitted symbolic regression to a simplified and potentially rounded mathematical expression.
    Warning: eval is used in this function, thus it should not be used on unsanitized input (see
    https://docs.sympy.org/latest/modules/core.html?highlight=eval#module-sympy.core.sympify).

    Parameters
    ----------
    symb: Fitted symbolic regressor to find a simplified expression for.
    n_decimals: If set, round floats in the expression to this number of decimals.

    Returns
    -------
    symb_conv: Converted mathematical expression.
    """
    if isinstance(symb, SymbolicRegressor):
        symb = str(symb._program)
    elif isinstance(symb, SymbolicMetaModelWrapper):
        symb = symb.expression()
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

    x, X0 = sympy.symbols("x X0")
    symb_conv = sympy.simplify(sympy.sympify(symb, locals=converter))
    symb_conv = symb_conv.subs(X0, x)
    if n_decimals:
        symb_conv = symb_conv.evalf(n_decimals)

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


def plot_symb(
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
    Create a plot showing the training points from SMAC and random sampling, as well as the true function and the
    two functions fitted by symbolic regression.
    """
    X_train_smac, y_train_smac = sort(X_train_smac, y_train_smac)
    X_train_rand, y_train_rand = sort(X_train_rand, y_train_rand)
    X_test, y_test = sort(X_test, y_test)
    fig = plt.figure(figsize=(8, 5))
    plt.title(f"{function.expression}")
    plt.plot(
        X_test,
        y_test,
        color="C4",
        linewidth=3,
        label=f"True function: {function.expression}",
    )

    for model_name in symbolic_models:
        symbolic_model = symbolic_models[model_name]

        conv = convert_symb(symbolic_model, n_decimals=3)
        if len(str(conv)) < 30:
            label = f"Predicted function ({model_name}): {conv}"
        else:
            label = f"Predicted function ({model_name})"
        plt.plot(
            X_test,
            symbolic_model.predict(X_test),
            #color="C1",
            linewidth=2,
            #linestyle="--",
            label=label
        )

    # conv_rand = convert_symb(symb_rand, n_decimals=3)
    # if len(str(conv_rand)) < 30:
    #     label_rand = f"Predicted function (random sampling): {conv_rand}"
    # else:
    #     label_rand = "Predicted function (random sampling)"
    # plt.plot(
    #     X_test,
    #     symb_rand.predict(X_test),
    #     color="C2",
    #     linewidth=2,
    #     linestyle="-.",
    #     label=label_rand,
    # )
    plt.scatter(
        X_train_smac,
        y_train_smac,
        color="C5",
        zorder=2,
        marker="^",
        s=40,
        label=f"Train points (SMAC sampling)",
    )
    plt.scatter(
        X_train_rand,
        y_train_rand,
        color="C3",
        zorder=2,
        marker="P",
        s=40,
        label="Train points (random sampling)",
    )
    epsilon = (X_test.max() - X_test.min()) / 100
    plt.xlim(X_test.min() - epsilon, X_test.max() + epsilon)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f"{plot_dir}/{function.name.lower().replace(' ', '_')}", dpi=300)
    else:
        plt.show()
    return fig


def write_dict_to_cfg_file(dictionary: dict, target_file_path: str):
    parser = cfgparse.ConfigParser()
    section = 'symbolic_regression'
    parser.add_section(section)

    for key in dictionary.keys():
        parser.set(section, key, str(dictionary[key]))
    with open(target_file_path, 'w') as f:
        parser.write(f)
