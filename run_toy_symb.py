import logging
from os import path
import numpy as np
import pandas as pd
from smac import BlackBoxFacade
from gplearn.genetic import SymbolicRegressor

from symbolic_meta_model_wrapper import (
    SymbolicMetaModelWrapper,
    SymbolicPursuitModelWrapper,
)
from symb_reg_utils import get_function_set
from utils import (
    get_output_dirs,
    convert_symb,
    append_scores,
    plot_symb1d,
    plot_symb2d,
    write_dict_to_cfg_file,
)
from smac_utils import run_smac_optimization
from functions import get_functions1d, get_functions2d


if __name__ == "__main__":
    seed = 42
    n_smac_samples = 20
    n_test_samples = 100
    n_dim = 2
    metric = "mean absolute error"
    symb_reg = True
    symb_meta = False
    symb_purs = False

    assert n_dim in (1, 2), f"Currently, n_dim can only be in (1,2), got: {n_dim}."

    np.random.seed(seed)

    functions = get_functions2d() if n_dim == 2 else get_functions1d()

    run_dir, res_dir, plot_dir = get_output_dirs()

    # setup logging
    logger = logging.getLogger(__name__)

    df_scores_symb_reg = pd.DataFrame() if symb_reg else None
    df_scores_symb_meta = pd.DataFrame() if symb_meta else None
    df_scores_symb_purs = pd.DataFrame() if symb_purs else None
    df_expr = pd.DataFrame()

    for function in functions:
        logger.info(f"Run SMAC for: {function.name}: {function.expression}")

        # get train samples for SR from SMAC sampling
        samples_smac, _ = run_smac_optimization(
            configspace=function.cs,
            facade=BlackBoxFacade,
            target_function=function.smac_apply,
            function_name=function.name.lower().replace(" ", "_"),
            n_eval=n_smac_samples,
            run_dir=run_dir,
            seed=seed,
        )

        X_train_smac = samples_smac
        y_train_smac = function.apply(X_train_smac)

        # get train samples for SR from random sampling
        X_train_rand = function.cs.sample_configuration(size=n_smac_samples)
        X_train_rand = np.array(
            [list(i.get_dictionary().values()) for i in X_train_rand]
        ).T
        y_train_rand = function.apply(X_train_rand)

        # get test samples for SR
        parameters = function.cs.get_hyperparameters()
        if n_dim == 2:
            test_lower, test_upper = [], []
            for i in range(2):
                lower = parameters[i].lower
                upper = parameters[i].upper
                test_lower.append(
                    lower + 0.5 * (upper - lower) / int(np.sqrt(n_test_samples))
                )
                test_upper.append(
                    upper - (0.5 * (upper - lower) / int(np.sqrt(n_test_samples)))
                )
            X_test = np.array(
                np.meshgrid(
                    np.linspace(
                        test_lower[0], test_upper[0], int(np.sqrt(n_test_samples))
                    ),
                    np.linspace(
                        test_lower[1], test_upper[1], int(np.sqrt(n_test_samples))
                    ),
                )
            )
        else:
            X_test = np.linspace(
                parameters[0].lower, parameters[0].upper, n_test_samples
            )
        y_test = function.apply(X_test)

        if n_dim == 1:
            y_train_smac, y_train_rand, y_test = (
                y_train_smac.reshape(-1),
                y_train_rand.reshape(-1),
                y_test.reshape(-1),
            )

        logger.info(f"Fit Symbolic Models for: {function.name}: {function.expression}")

        # Create a safe exp function which does not cause problems
        def exp(x):
            with np.errstate(all="ignore"):
                # TODO: We maybe want to set a larger upper limit
                max_value = np.full(shape=x.shape, fill_value=100000)
                return np.minimum(np.exp(x), max_value)

        symbolic_models = {}

        if symb_reg:
            # TODO: log symb regression logs?
            symb_params = dict(
                population_size=5000,
                generations=50,
                stopping_criteria=0.001,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                parsimony_coefficient=0.01,
                function_set=get_function_set(),
                metric=metric,
                random_state=0,
                verbose=0,
            )

            write_dict_to_cfg_file(
                dictionary=symb_params,
                target_file_path=path.join(run_dir, "symbolic_regression_params.cfg"),
            )

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
                function.expression,
                symb_smac,
                symb_rand,
                X_train_smac.T,
                y_train_smac,
                X_train_rand.T,
                y_train_rand,
                X_test.reshape(n_dim, -1).T,
                y_test.reshape(-1),
            )
            df_scores_symb_reg.to_csv(f"{res_dir}/scores_symb_reg.csv")

        if symb_meta:
            # run symbolic metamodels on SMAC samples
            symb_meta_smac = SymbolicMetaModelWrapper()
            symb_meta_smac.fit(X_train_smac.T, y_train_smac)
            # or run symbolic metaexpressions on SMAC samples
            # symb_meta_smac, _ = get_symbolic_model(function.apply, X_train_smac)
            # symb_meta_smac = SymbolicMetaExpressionWrapper(symb_meta_smac)

            # run symbolic metamodels on random samples
            symb_meta_rand = SymbolicMetaModelWrapper()
            symb_meta_rand.fit(X_train_rand.T, y_train_rand)
            # or run symbolic metaexpressions on SMAC samples
            # symb_meta_rand, _ = get_symbolic_model(function.apply, X_train_rand)
            # symb_meta_rand = SymbolicMetaExpressionWrapper(symb_meta_rand)

            symbolic_models["Meta-smac"] = symb_meta_smac
            symbolic_models["Meta-rand"] = symb_meta_rand

            # write results to csv files
            df_scores_symb_meta = append_scores(
                df_scores_symb_meta,
                function.expression,
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

        if symb_purs:
            purs_params = dict(
                loss_tol=1.0e-3,
                ratio_tol=0.9,
                patience=10,
                maxiter=20,
                eps=1.0e-5,
                random_seed=42,
                task_type="regression",
            )

            write_dict_to_cfg_file(
                dictionary=purs_params,
                target_file_path=path.join(run_dir, "symbolic_pursuit_params.cfg"),
            )

            # run symbolic pursuit models on SMAC samples
            symb_purs_smac = SymbolicPursuitModelWrapper(**purs_params)
            symb_purs_smac.fit(X_train_smac.T, y_train_smac)

            # run symbolic pursuit models on random samples
            symb_purs_rand = SymbolicPursuitModelWrapper(**purs_params)
            symb_purs_rand.fit(X_train_rand.T, y_train_rand)

            symbolic_models["Pursuit-smac"] = symb_purs_smac
            symbolic_models["Pursuit-rand"] = symb_purs_rand

            # write results to csv files
            df_scores_symb_purs = append_scores(
                df_scores_symb_purs,
                function.expression,
                symb_purs_smac,
                symb_purs_rand,
                X_train_smac.T,
                y_train_smac,
                X_train_rand.T,
                y_train_rand,
                X_test.reshape(n_dim, n_test_samples).T,
                y_test.reshape(n_test_samples),
            )
            df_scores_symb_purs.to_csv(f"{res_dir}/scores_symb_purs.csv")

        df_expr[function.expression] = {
            k: convert_symb(v, n_dim=n_dim, n_decimals=3)
            for k, v in symbolic_models.items()
        }
        df_expr.to_csv(f"{res_dir}/functions.csv")

        # plot results
        if n_dim == 1:
            plot = plot_symb1d(
                X_train_smac=X_train_smac.T,
                y_train_smac=y_train_smac,
                X_train_rand=X_train_rand.T,
                y_train_rand=y_train_rand,
                X_test=X_test,
                y_test=y_test,
                xlabel="x",
                ylabel="f(x)",
                symbolic_models=symbolic_models,
                function_name=function.name.lower().replace(" ", "_"),
                function_expression=function.expression,
                plot_dir=plot_dir,
            )
        else:
            plot = plot_symb2d(
                X_train_smac=X_train_smac,
                X_train_rand=X_train_rand,
                X_test=X_test,
                y_test=y_test,
                symbolic_models=symbolic_models,
                parameters=function.cs.get_hyperparameters(),
                function_name=function.name.lower().replace(" ", "_"),
                function_expression=function.expression,
                plot_dir=plot_dir,
            )
