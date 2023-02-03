import logging
from os import path
import numpy as np
import pandas as pd
from functools import partial
from gplearn.genetic import SymbolicRegressor
from smac import HyperparameterOptimizationFacade
from ConfigSpace import Configuration, UniformIntegerHyperparameter, UniformFloatHyperparameter

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
from model_wrapper import SVM, MLP


if __name__ == "__main__":
    seed = 42
    n_smac_samples = 20
    n_test_samples = 100
    model = "SVM"
    symb_reg = True
    symb_meta = False
    symb_purs = False

    compare_on_test = True

    if model == "MLP":
        classifier = MLP(
            optimize_n_neurons=True,
            optimize_batch_size=True,
            optimize_learning_rate_init=False,
            seed=seed,
        )
    elif model == "SVM":
        classifier = SVM(
            optimize_C=True,
            optimize_degree=True,
            optimize_coef=False,
            optimize_gamma=False,
        )
    else:
        print(f"Unknown model: {model}")
        classifier = None

    np.random.seed(seed)

    run_dir, res_dir, plot_dir = get_output_dirs()

    # setup logging
    logger = logging.getLogger(__name__)

    optimized_parameters = classifier.configspace.get_hyperparameters()

    df_scores_symb_reg = pd.DataFrame() if symb_reg else None
    df_scores_symb_meta = pd.DataFrame() if symb_meta else None
    df_scores_symb_purs = pd.DataFrame() if symb_purs else None
    df_expr = pd.DataFrame()

    logger.info(f"Run SMAC to sample configs and train {model}.")

    X_train_smac, y_train_smac, smac_facade = run_smac_optimization(
        configspace=classifier.configspace,
        facade=HyperparameterOptimizationFacade,
        target_function=classifier.train,
        function_name=model,
        n_eval=n_smac_samples,
        run_dir=run_dir,
        seed=seed,
    )
    X_train_smac = X_train_smac.astype(float)

    logger.info(f"Create grid configs for testing and train {model}.")

    # get test samples for SR
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
            int_spacing = np.unique(([int(i) for i in param_space] + [optimized_parameters[i].upper]))
            X_test_dimensions.append(int_spacing)
        else:
            X_test_dimensions.append(param_space)

    param_dict = {}
    if len(optimized_parameters) == 1:
        X_test = X_test_dimensions[0]
        y_test = np.zeros(len(X_test_dimensions[0]))
        for n in range(len(X_test_dimensions[0])):
            param_dict[optimized_parameters[0].name] = X_test[n]
            conf = Configuration(
                configuration_space=classifier.configspace, values=param_dict
            )
            y_test[n] = classifier.train(config=conf, seed=seed)
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
                conf = Configuration(
                    configuration_space=classifier.configspace, values=param_dict
                )
                y_test[n, m] = classifier.train(config=conf, seed=seed)
    else:
        X_test = None
        y_test = None
        print("Not yet supported.")

    if compare_on_test:
        X_train_compare = X_test.copy().reshape(len(optimized_parameters), -1)
        y_train_compare = y_test.copy().reshape(-1)
    else:
        logger.info(f"Sample random configs and train {model}.")

        # get train samples for SR from random sampling
        X_train_rand = classifier.configspace.sample_configuration(size=n_smac_samples)
        y_train_rand = np.array(
            [classifier.train(config=x, seed=seed) for x in X_train_rand]
        )
        X_train_rand = np.array(
            [list(i.get_dictionary().values()) for i in X_train_rand]
        ).T.astype(float)
        X_train_compare = X_train_rand.copy()
        y_train_compare = y_train_rand.copy()

    # log transform values of parameters that were log-sampled before training the symbolic models
    for i in range(len(optimized_parameters)):
        if optimized_parameters[i].log:
            X_train_smac[i, :] = np.log(X_train_smac[i, :])
            X_train_compare[i, :] = np.log(X_train_compare[i, :])
            X_test[i, :] = np.log(X_test[i, :])

    logger.info(f"Fit Symbolic Models for {model}.")

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
            metric="mse", #"mean absolute error",
            random_state=0,
            verbose=1,
        )

        write_dict_to_cfg_file(
            dictionary=symb_params,
            target_file_path=path.join(run_dir, "symbolic_regression_params.cfg"),
        )

        # run SR on SMAC samples
        symb_smac = SymbolicRegressor(**symb_params)
        symb_smac.fit(X_train_smac.T, y_train_smac)
        symbolic_models["Symb-smac"] = symb_smac

        # run SR on compare samples (either random samples or test grid samples)
        symb_compare = SymbolicRegressor(**symb_params)
        symb_compare.fit(X_train_compare.T, y_train_compare)
        if compare_on_test:
            symbolic_models["Symb-test"] = symb_compare
            comp_postfix = "test"
        else:
            symbolic_models["Symb-rand"] = symb_compare
            comp_postfix = "rand"

        # write results to csv files
        df_scores_symb_reg = append_scores(
            df_scores_symb_reg,
            model,
            symb_smac,
            symb_compare,
            X_train_smac.T,
            y_train_smac,
            X_train_compare.T,
            y_train_compare,
            comp_postfix,
            X_test.reshape(len(optimized_parameters), -1).T,
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
        symb_meta_rand.fit(X_train_compare.T, y_train_compare)
        # or run symbolic metaexpressions on SMAC samples
        # symb_meta_rand, _ = get_symbolic_model(function.apply, X_train_compare)
        # symb_meta_rand = SymbolicMetaExpressionWrapper(symb_meta_rand)

        symbolic_models["Meta-smac"] = symb_meta_smac
        symbolic_models["Meta-rand"] = symb_meta_rand

        # write results to csv files
        df_scores_symb_meta = append_scores(
            df_scores_symb_meta,
            model,
            symb_meta_smac,
            symb_meta_rand,
            X_train_smac.T,
            y_train_smac,
            X_train_compare.T,
            y_train_compare,
            comp_postfix,
            X_test.reshape(len(optimized_parameters), -1).T,
            y_test.reshape(-1),
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
        symb_purs_rand.fit(X_train_compare.T, y_train_compare)

        symbolic_models["Pursuit-smac"] = symb_purs_smac
        symbolic_models["Pursuit-rand"] = symb_purs_rand

        # write results to csv files
        df_scores_symb_purs = append_scores(
            df_scores_symb_purs,
            model,
            symb_purs_smac,
            symb_purs_rand,
            X_train_smac.T,
            y_train_smac,
            X_train_compare.T,
            y_train_compare,
            comp_postfix,
            X_test.reshape(len(optimized_parameters), -1).T,
            y_test.reshape(-1),
        )
        df_scores_symb_purs.to_csv(f"{res_dir}/scores_symb_purs.csv")

    df_expr[model] = {
        k: convert_symb(v, n_dim=len(optimized_parameters), n_decimals=3)
        for k, v in symbolic_models.items()
    }
    df_expr.to_csv(f"{res_dir}/functions.csv")

    # plot results
    if len(optimized_parameters) == 1:
        param = optimized_parameters[0]
        plot = plot_symb1d(
            X_train_smac=X_train_smac.T,
            y_train_smac=y_train_smac,
            X_train_rand=X_train_compare.T,
            y_train_rand=y_train_compare,
            X_test=X_test.T,
            y_test=y_test,
            xlabel=f"log({param.name})" if param.log else param.name,
            ylabel="Cost",
            symbolic_models=symbolic_models,
            function_name=model,
            xmin=np.log(param.lower) if param.log else param.lower,
            xmax=np.log(param.upper) if param.log else param.upper,
            plot_dir=plot_dir,
        )
    elif len(optimized_parameters) == 2:
        plot = plot_symb2d(
            X_train_smac=X_train_smac,
            X_train_compare=X_train_compare,
            X_test=X_test,
            y_test=y_test,
            function_name=model,
            metric_name="Cost",
            symbolic_models=symbolic_models,
            parameters=optimized_parameters,
            plot_dir=plot_dir,
        )
