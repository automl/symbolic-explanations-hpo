import logging
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor

from utils import get_output_dirs, convert_symb, append_scores, plot_symb
from smac_utils import run_smac_optimization
from functions import get_functions


if __name__ == "__main__":
    x_min = 0.01
    x_max = 1.00
    n_eval = 10

    functions = get_functions()
    run_dir, res_dir, plot_dir = get_output_dirs()

    # setup logging
    logger = logging.getLogger(__name__)

    df_scores, df_expr = pd.DataFrame(), pd.DataFrame()

    for function in functions:
        # get train samples for SR from SMAC sampling
        samples_smac, _ = run_smac_optimization(
            hp_space={"x": (x_min, x_max)},
            function=function,
            n_eval=n_eval,
            run_dir=run_dir,
        )

        logger.info(
            f"Fit Symbolic Regression for: {function.name}: {function.expression}"
        )

        X_train_smac = samples_smac.reshape(-1, 1)
        y_train_smac = function.apply(X_train_smac).reshape(-1)

        # get train samples for SR from random sampling
        X_train_rand = np.random.uniform(x_min, x_max, size=n_eval).reshape(-1, 1)
        y_train_rand = function.apply(X_train_rand).reshape(-1)

        # get test samples for SR
        X_test = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_test = function.apply(X_test).reshape(-1)

        # SR settings
        function_set = ["add", "sub", "mul", "div", "sqrt", "log", "sin", "cos"]
        # TODO: log symb regression logs?
        symb_params = dict(
            population_size=5000,
            generations=5,
            stopping_criteria=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            parsimony_coefficient=0.01,
            function_set=function_set,
            metric="mean absolute error",
            random_state=0,
            verbose=0,
        )

        # run SR on SMAC samples
        symb_smac = SymbolicRegressor(**symb_params)
        symb_smac.fit(X_train_smac, y_train_smac)

        # run SR on random samples
        symb_rand = SymbolicRegressor(**symb_params)
        symb_rand.fit(X_train_rand, y_train_rand)

        # write results to csv files
        df_scores = append_scores(
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
        )
        df_expr[function.expression] = {
            "symb_smac": convert_symb(symb_smac),
            "symb_rand": convert_symb(symb_rand),
        }
        df_scores.to_csv(f"{res_dir}/scores.csv")
        df_expr.to_csv(f"{res_dir}/functions.csv")

        # plot results
        plot = plot_symb(
            X_train_smac,
            y_train_smac,
            X_train_rand,
            y_train_rand,
            X_test,
            y_test,
            symb_smac,
            symb_rand,
            function,
            plot_dir=plot_dir,
        )
