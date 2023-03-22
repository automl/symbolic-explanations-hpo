import os
import time

import pandas as pd
import sympy
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from functools import partial
from gplearn.genetic import SymbolicRegressor
from gplearn import functions
from symbolic_meta_model_wrapper import (
    SymbolicMetaModelWrapper,
    SymbolicPursuitModelWrapper,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import configparser as cfgparse
from ConfigSpace import Configuration, UniformIntegerHyperparameter
from smac.runhistory.encoder.encoder import convert_configurations_to_array
from utils.functions_utils import get_functions2d, NamedFunction
from utils.model_utils import get_models
from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark

plt.style.use("tableau-colorblind10")


def get_output_dirs() -> [str, str, str]:
    """
    Create directory for current run as well as a plot and result subdirectory.
    """
    if not os.path.exists("../runs"):
        os.makedirs("../runs")
    run_dir = f"runs/run_{time.strftime('%Y%m%d-%H%M%S')}"
    res_dir = f"{run_dir}/results"
    plot_dir = f"{run_dir}/plots"
    os.makedirs(run_dir)
    os.makedirs(res_dir)
    os.makedirs(plot_dir)
    return run_dir, res_dir, plot_dir


# def get_run_config(job_id, n_optimized_params, toy_problems=False):
#     run_configs = []
#     if toy_problems:
#         models = get_functions2d()
#     else:
#         models = get_models()
#     for model in models:
#         if isinstance(model, NamedFunction):
#             run_configs.append({"model": model, "data_set_name": None})
#         else:
#             hyperparams = get_hyperparams(model_name=model)
#             hp_comb = combinations(hyperparams, 2)
#             for hp_conf in hp_comb:
#                 for ds in data_sets:
#                     run_configs.append({"model": model, hp_conf[0]: True, hp_conf[1]: True, "data_set_name": ds})
#     return run_configs[int(job_id)]


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

        # sqrt is protected function in gplearn, always returning sqrt(abs(x))
        sqrt_pos = []
        prev_sqrt_inserts = 0
        for i, f in enumerate(symb._program.program):
            if isinstance(f, functions._Function) and f.name == "sqrt":
                sqrt_pos.append(i)
        for i in sqrt_pos:
            symb._program.program.insert(i + prev_sqrt_inserts + 1, functions.abs1)
            prev_sqrt_inserts += 1

        # log is protected function in gplearn, always returning sqrt(abs(x))
        log_pos = []
        prev_log_inserts = 0
        for i, f in enumerate(symb._program.program):
            if isinstance(f, functions._Function) and f.name == "log":
                log_pos.append(i)
        for i in log_pos:
            symb._program.program.insert(i + prev_log_inserts + 1, functions.abs1)
            prev_log_inserts += 1

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
        "pow": lambda x, y: x**y
    }

    if symb._program._length > 300:
        print(
            f"Expression of length {symb._program._length} too long to convert, return raw string."
        )
        return symb_str

    symb_conv = sympy.sympify(symb_str.replace("[", "").replace("]", ""), locals=converter)
    if n_dim == 1:
        x, X0 = sympy.symbols("x X0")
        symb_conv = symb_conv.subs(X0, x)
    if n_dim == 2:
        X0, X1 = sympy.symbols("X0 X1", real=True)
        symb_conv = symb_conv.subs(X0, X1)
    symb_simpl = sympy.simplify(symb_conv)

    if isinstance(symb, SymbolicPursuitModelWrapper):
        proj = symb.metamodel.get_projections()
        for i, p in enumerate(proj):
            for a in sympy.preorder_traversal(symb_simpl):
                if isinstance(a, sympy.core.Symbol) and str(a) == f"P{i+1}":
                    symb_simpl = symb_simpl.subs(a, p)
    if n_decimals:
        # Make sure also floats deeper in the expression tree are rounded
        for a in sympy.preorder_traversal(symb_simpl):
            if isinstance(a, sympy.core.numbers.Float):
                symb_simpl = symb_simpl.subs(a, round(a, n_decimals))

    return symb_simpl


def append_scores(
    df_scores,
    col_name,
    symb_smac,
    symb_comp,
    X_train_smac,
    y_train_smac,
    X_train_comp,
    y_train_comp,
    comp_postfix,
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
        f"mae_train_{comp_postfix}": mean_absolute_error(
            y_train_comp, symb_comp.predict(X_train_comp)
        ),
        "mae_test_smac": mean_absolute_error(y_test, symb_smac.predict(X_test)),
        f"mae_test_{comp_postfix}": mean_absolute_error(
            y_test, symb_comp.predict(X_test)
        ),
        "mse_train_smac": mean_squared_error(
            y_train_smac, symb_smac.predict(X_train_smac)
        ),
        f"mse_train_{comp_postfix}": mean_squared_error(
            y_train_comp, symb_comp.predict(X_train_comp)
        ),
        "mse_test_smac": mean_squared_error(y_test, symb_smac.predict(X_test)),
        f"mse_test_{comp_postfix}": mean_squared_error(
            y_test, symb_comp.predict(X_test)
        ),
        "r2_train_smac": r2_score(y_train_smac, symb_smac.predict(X_train_smac)),
        f"r2_train_{comp_postfix}": r2_score(
            y_train_comp, symb_comp.predict(X_train_comp)
        ),
        "r2_test_smac": r2_score(y_test, symb_smac.predict(X_test)),
        f"r2_test_{comp_postfix}": r2_score(y_test, symb_comp.predict(X_test)),
    }
    return df_scores


def get_scores(
    y_train,
    pred_train,
    y_test,
    pred_test
):
    """
    Get scores.
    """
    df_scores = pd.DataFrame.from_dict({
        "mae_train": [mean_absolute_error(y_train, pred_train)],
        "mae_test": [mean_absolute_error(y_test, pred_test)],
        "mse_train": [mean_squared_error(y_train, pred_train)],
        "mse_test": [mean_squared_error(y_test, pred_test)],
        "r2_train": [r2_score(y_train, pred_train)],
        "r2_test": [r2_score(y_test, pred_test)],
        "kt_train": [kendalltau(y_train, pred_train)[0]],
        "kt_test": [kendalltau(y_test, pred_test)[0]],
        "kt_p_train": [kendalltau(y_train, pred_train)[1]],
        "kt_p_test": [kendalltau(y_test, pred_test)[1]],
    })
    return df_scores


def get_surrogate_predictions(X, cs, surrogate_model):
    y_surrogate = []
    optimized_parameters = cs.get_hyperparameters()
    for i in range(X.shape[0]):
        x0 = int(X[i][0]) if isinstance(optimized_parameters[0], UniformIntegerHyperparameter) else X[i][0]
        x1 = int(X[i][1]) if isinstance(optimized_parameters[1], UniformIntegerHyperparameter) else X[i][1]
        conf = Configuration(
            configuration_space=cs,
            values={
                optimized_parameters[0].name: x0,
                optimized_parameters[1].name: x1
            },
        )
        y_surrogate.append(surrogate_model.predict(convert_configurations_to_array([conf]))[0][0][0])
    return y_surrogate


def write_dict_to_cfg_file(dictionary: dict, target_file_path: str):
    parser = cfgparse.ConfigParser()
    section = "symbolic_regression"
    parser.add_section(section)

    for key in dictionary.keys():
        parser.set(section, key, str(dictionary[key]))
    with open(target_file_path, "w") as f:
        parser.write(f)


def get_hpo_test_data(classifier, optimized_parameters, n_test_samples, n_test_eval=5, return_x=False):
    # Get test grid configurations
    X_test_dimensions = []
    n_test_steps = (
        int(np.sqrt(n_test_samples))
        if len(optimized_parameters) == 2
        else n_test_samples
        if len(optimized_parameters) == 1
        else None
    )
    for i in range(len(optimized_parameters)):
        space = (
            partial(np.logspace, base=np.e)
            if optimized_parameters[i].log
            else np.linspace
        )
        if optimized_parameters[i].log:
            lower = np.log(optimized_parameters[i].lower)
            upper = np.log(optimized_parameters[i].upper)
        else:
            lower = optimized_parameters[i].lower
            upper = optimized_parameters[i].upper
        param_space = space(
            lower + 0.5 * (upper - lower) / n_test_steps,
            upper - (0.5 * (upper - lower) / n_test_steps),
            n_test_steps,
        )
        if isinstance(optimized_parameters[i], UniformIntegerHyperparameter):
            int_spacing = np.unique(
                ([int(i) for i in param_space])
            )
            if optimized_parameters[i].upper not in int_spacing:
                int_spacing = np.append(int_spacing, optimized_parameters[i].upper)
            X_test_dimensions.append(int_spacing)
        else:
            X_test_dimensions.append(param_space)

    param_dict = {}
    if len(optimized_parameters) == 1:
        X_test = X_test_dimensions[0]
        if return_x:
            return X_test
        y_test = np.zeros(len(X_test_dimensions[0]))
        for n in range(len(X_test_dimensions[0])):
            param_dict[optimized_parameters[0].name] = X_test[n]
            if isinstance(classifier, MLBenchmark):
                cs = classifier.configuration_space
            else:
                cs = classifier.configspace
            conf = Configuration(
                configuration_space=cs, values=param_dict
            )
            for i in range(n_test_eval):
                seed = i * 3
                if isinstance(classifier, MLBenchmark):
                    y_test[n] += classifier.objective_function(conf.get_dictionary(), seed=seed)
                else:
                    y_test[n] += classifier.train(config=conf, seed=seed)
            y_test[n] = y_test[n] / n_test_eval
        X_test, y_test = X_test.astype(float).reshape(
            1, X_test.shape[0]
        ), y_test.reshape(-1)
    elif len(optimized_parameters) == 2:
        X_test = np.array(
            np.meshgrid(
                X_test_dimensions[0],
                X_test_dimensions[1],
            )
        ).astype(float)
        if return_x:
            return X_test

        # Train model to get actual loss for each test config
        y_test = np.zeros((X_test.shape[1], X_test.shape[2]))
        for n in range(X_test.shape[1]):
            for m in range(X_test.shape[2]):
                for i, param in enumerate(optimized_parameters):
                    if isinstance(
                        optimized_parameters[i], UniformIntegerHyperparameter
                    ):
                        param_dict[optimized_parameters[i].name] = int(X_test[i, n, m])
                    else:
                        param_dict[optimized_parameters[i].name] = X_test[i, n, m]
                if isinstance(classifier, MLBenchmark):
                    cs = classifier.configuration_space
                else:
                    cs = classifier.configspace
                conf = Configuration(
                    configuration_space=cs, values=param_dict
                )
                for i in range(n_test_eval):
                    seed = i * 3
                    if isinstance(classifier, MLBenchmark):
                        y_test[n, m] += classifier.objective_function(configuration=conf.get_dictionary(), seed=seed)[
                            "function_value"]
                    else:
                        y_test[n, m] += classifier.train(config=conf, seed=seed)
                y_test[n, m] = y_test[n, m] / n_test_eval
    else:
        X_test = None
        y_test = None
        print("Not yet supported.")

    return X_test, y_test


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
        axes[0].set_title(f"True: {function_expression}", fontsize=LABEL_SIZE)
    else:
        axes[0].set_title(f"{function_name}", fontsize=LABEL_SIZE)
    axes[0].set_xlabel(X0_name, fontsize=LABEL_SIZE)
    axes[0].set_ylabel(X1_name, fontsize=LABEL_SIZE)
    axes[0].set_xticks(dim_x)
    axes[0].set_yticks(dim_y)
    axes[0].set_xlim(X0_lower, X0_upper)
    axes[0].set_ylim(X1_lower, X1_upper)
    axes[0].tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    axes[0].grid(alpha=0)

    if not use_same_scale:
        cbar = fig.colorbar(im, ax=axes[0])
        cbar.set_label(r'True $\mathcal{L}$', fontsize=LABEL_SIZE, rotation=270, labelpad=10)
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
        axes[i + 1].set_title(f"{label}", fontsize=LABEL_SIZE)
        axes[i + 1].set_xlabel(X0_name, fontsize=LABEL_SIZE)
        axes[i + 1].set_ylabel(X1_name, fontsize=LABEL_SIZE)
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
                label="Train points",
            )
    if not use_same_scale:
        cbar = fig.colorbar(im, ax=axes[1:], shrink=0.4)
        if metric_name:
            cbar.set_label(metric_name, fontsize=LABEL_SIZE, rotation=270, labelpad=15)
        else:
            cbar.set_label("f(X0, X1)", fontsize=LABEL_SIZE, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=LABEL_SIZE)

    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower right", fontsize=LABEL_SIZE, framealpha=0.0
    )
    leg.get_frame().set_linewidth(0.0)
    if use_same_scale:
        cbar = fig.colorbar(im, ax=axes, shrink=0.4)
        if metric_name:
            cbar.set_label(metric_name, fontsize=LABEL_SIZE, rotation=270, labelpad=15)
        else:
            cbar.set_label("f(X0, X1)", fontsize=LABEL_SIZE, rotation=270, labelpad=15)
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
        axes[0].set_title(f"{function_name}", fontsize=LABEL_SIZE)
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
