import logging
from os import path
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from symbolic_metamodeling.symbolic_meta_model_wrapper import SymbolicMetaModelWrapper, SymbolicMetaExpressionWrapper
from symbolic_metamodeling.pysymbolic.algorithms.symbolic_expressions import get_symbolic_model
from gplearn.functions import make_function

from utils import get_output_dirs, convert_symb, append_scores, plot_symb1d, plot_symb2d, write_dict_to_cfg_file
from smac_utils import run_smac_optimization
from functions import get_functions1d, get_functions2d


if __name__ == "__main__":
    x_min = 0.01
    x_max = 1.00
    n_smac_samples = 10
    n_test_samples = 100
    n_dim = 2
    symb_reg = True
    symb_meta = False

    assert n_dim in (1, 2), f"Currently, n_dim can only be in (1,2), got: {n_dim}."
    assert not (symb_meta and n_dim == 2), f"Currently, Symbolic Metamodels can only be used with n_dim=1."

    functions = get_functions2d() if n_dim == 2 else get_functions1d()

    run_dir, res_dir, plot_dir = get_output_dirs()

    # setup logging
    logger = logging.getLogger(__name__)

    df_scores_symb_reg = pd.DataFrame() if symb_reg else None
    df_scores_symb_meta = pd.DataFrame() if symb_meta else None
    df_expr = pd.DataFrame()

    for function in functions:
        # get train samples for SR from SMAC sampling
        hp_space = {f"x{i}": (x_min, x_max) for i in range(n_dim)} if n_dim > 1 else {"x": (x_min, x_max)}
        samples_smac, _ = run_smac_optimization(
            hp_space=hp_space,
            function=function,
            n_eval=n_smac_samples,
            run_dir=run_dir,
        )

        logger.info(
            f"Fit Symbolic Regression for: {function.name}: {function.expression}"
        )

        X_train_smac = samples_smac
        y_train_smac = function.apply(X_train_smac)

        # get train samples for SR from random sampling
        X_train_rand = np.random.uniform(x_min, x_max, size=(n_dim, n_smac_samples))
        y_train_rand = function.apply(X_train_rand)#.reshape(-1)

        # get test samples for SR
        if n_dim == 2:
            X_test = np.array(np.meshgrid(np.linspace(x_min, x_max, int(np.sqrt(n_test_samples))),
                                          np.linspace(x_min, x_max, int(np.sqrt(n_test_samples)))))
        else:
            X_test = np.linspace(x_min, x_max, n_test_samples)
        y_test = function.apply(X_test)#.reshape(-1)

        if n_dim == 1:
            y_train_smac, y_train_rand, y_test = y_train_smac.reshape(-1), y_train_rand.reshape(-1), y_test.reshape(-1)

        # Create a safe exp function which does not cause problems
        def exp(x):
            with np.errstate(all='ignore'):
                # TODO: We maybe want to set a larger upper limit
                max_value = np.full(shape=x.shape, fill_value=100000)
                return np.minimum(np.exp(x), max_value)

        symbolic_models = {}

        if symb_reg:
            # SR settings
            exp_func = make_function(function=exp, arity=1, name="exp")
            function_set = ["add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", exp_func]
            # TODO: log symb regression logs?
            symb_params = dict(
                population_size=1000,
                generations=50,
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
                verbose=0
            )

            write_dict_to_cfg_file(dictionary=symb_params, target_file_path=path.join(run_dir, 'symbolic_regression_params.cfg'))

            # run SR on SMAC samples
            symb_smac = SymbolicRegressor(**symb_params)
            symb_smac.fit(X_train_smac.T, y_train_smac)

            # run SR on random samples
            symb_rand = SymbolicRegressor(**symb_params)
            symb_rand.fit(X_train_rand.T, y_train_rand)

            symbolic_models["Symb-smac"] = symb_smac
            symbolic_models["Symb-rand"] = symb_rand

            # write results to csv files
            df_scores_symb_reg = append_scores(
                df_scores_symb_reg,
                function,
                symb_smac,
                symb_rand,
                X_train_smac.T,
                y_train_smac,
                X_train_rand.T,
                y_train_rand,
                X_test.reshape(n_dim, n_test_samples).T,
                y_test.reshape(n_test_samples),
            )
            df_scores_symb_reg.to_csv(f"{res_dir}/scores_symb_reg.csv")

        if symb_meta:
            # run symbolic metamodels on SMAC samples
            symb_meta_smac = SymbolicMetaModelWrapper()
            symb_meta_smac.fit(X_train_smac.T, y_train_smac)
            # or run symbolic metaexpressions on SMAC samples
            #symb_meta_smac, _ = get_symbolic_model(function.apply, X_train_smac)
            #symb_meta_smac = SymbolicMetaExpressionWrapper(symb_meta_smac)

            # run symbolic metamodels on random samples
            symb_meta_rand = SymbolicMetaModelWrapper()
            symb_meta_rand.fit(X_train_rand.T, y_train_rand)
            # or run symbolic metaexpressions on SMAC samples
            #symb_meta_rand, _ = get_symbolic_model(function.apply, X_train_rand)
            #symb_meta_rand = SymbolicMetaExpressionWrapper(symb_meta_rand)

            symbolic_models["Meta-smac"] = symb_meta_smac
            symbolic_models["Meta-rand"] = symb_meta_rand

            # write results to csv files
            df_scores_symb_meta = append_scores(
                df_scores_symb_meta,
                function,
                symb_meta_smac,
                symb_meta_rand,
                X_train_smac.T,
                y_train_smac,
                X_train_rand.T,
                y_train_rand,
                X_test.reshape(n_dim, n_test_samples).T,
                y_test.reshape(n_test_samples),
            )
            df_scores_symb_meta.to_csv(f"{res_dir}/scores_symb_meta.csv")

        df_expr[function.expression] = {k: convert_symb(v, n_dim=n_dim, n_decimals=3) for k, v in
                                        symbolic_models.items()}
        df_expr.to_csv(f"{res_dir}/functions.csv")

        # plot results
        if n_dim == 1:
            plot = plot_symb1d(
                X_train_smac.T,
                y_train_smac,
                X_train_rand.T,
                y_train_rand,
                X_test,
                y_test,
                symbolic_models,
                function,
                plot_dir=plot_dir,
            )
        else:
            plot_symb2d(
                X_train_smac,
                X_train_rand,
                X_test,
                y_test,
                symbolic_models,
                function,
                plot_dir=plot_dir,
            )
