import pandas as pd
import sympy
import numpy as np
from scipy.stats import kendalltau
from functools import partial
from gplearn import functions
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import configparser as cfgparse
from ConfigSpace import Configuration, UniformIntegerHyperparameter
from smac.runhistory.encoder.encoder import convert_configurations_to_array
from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark


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


    converter = {
        "sub": lambda x, y: x - y,
        "div": lambda x, y: x / y,
        "mul": lambda x, y: x * y,
        "add": lambda x, y: x + y,
        "neg": lambda x: -x,
        "pow": lambda x, y: x**y
    }

    if symb._program.length_ > 300:
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

    if n_decimals:
        # Make sure also floats deeper in the expression tree are rounded
        for a in sympy.preorder_traversal(symb_simpl):
            if isinstance(a, sympy.core.numbers.Float):
                symb_simpl = symb_simpl.subs(a, round(a, n_decimals))

    return symb_simpl


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
