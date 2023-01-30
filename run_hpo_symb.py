import logging
from os import path
import numpy as np
import pandas as pd
from functools import partial
from gplearn.genetic import SymbolicRegressor
from symbolic_meta_model_wrapper import (
    SymbolicMetaModelWrapper, SymbolicPursuitModelWrapper
)
from symb_reg_utils import get_function_set

from ConfigSpace import Configuration, UniformIntegerHyperparameter

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
    symb_reg = True
    symb_meta = False
    symb_purs = False
    model = "MLP"
    classifier = None

    np.random.seed(seed)

    run_dir, res_dir, plot_dir = get_output_dirs()

    # setup logging
    logger = logging.getLogger(__name__)

    if model == "MLP":
        classifier = MLP(optimize_n_neurons=True, optimize_batch_size=True, seed=seed)
    elif model == "SVM":
        classifier = SVM()
    else:
        print(f"Unknown model: {model}")

    optimized_parameters = classifier.configspace.get_hyperparameters()

    logger.info(f"Run SMAC to sample configs and train {model}.")

    df_scores_symb_reg = pd.DataFrame() if symb_reg else None
    df_scores_symb_meta = pd.DataFrame() if symb_meta else None
    df_scores_symb_purs = pd.DataFrame() if symb_purs else None
    df_expr = pd.DataFrame()

    X_train_smac, y_train_smac = run_smac_optimization(
        configspace=classifier.configspace,
        target_function=classifier.train,
        function_name=model,
        n_eval=n_smac_samples,
        run_dir=run_dir,
        seed=seed
    )

   # initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    logger.info(
        f"Sample random configs and train {model}."
    )

    # get train samples for SR from random sampling
    X_train_rand = classifier.configspace.sample_configuration(size=n_smac_samples)
    y_train_rand = np.array([classifier.train(config=x, seed=seed) for x in X_train_rand])
    X_train_rand = np.array(
        [list(i.get_dictionary().values()) for i in X_train_rand]
    ).T

    logger.info(
        f"Sample configs for testing and train {model}."
    )

    # get test samples for SR
    if len(optimized_parameters) == 2:
        X_test_dimensions = []
        for i in range(len(optimized_parameters)):
            space = partial(np.logspace, base=np.e) if optimized_parameters[i].log else np.linspace
            if optimized_parameters[i].log:
                lower = np.log(optimized_parameters[i].lower)
                upper = np.log(optimized_parameters[i].upper)
            else:
                lower = optimized_parameters[i].lower
                upper = optimized_parameters[i].upper
            param_space = space(lower + 0.5 * (upper - lower) / int(np.sqrt(n_test_samples)),
                                upper - (0.5 * (upper - lower) / int(np.sqrt(n_test_samples))),
                                int(np.sqrt(n_test_samples))
                                )
            if isinstance(optimized_parameters[i], UniformIntegerHyperparameter):
                X_test_dimensions.append(np.unique(([int(i) for i in param_space])))
            else:
                X_test_dimensions.append(param_space)

        X_test = np.array(
            np.meshgrid(
                X_test_dimensions[0],
                X_test_dimensions[1],
            )
        ).astype(float)
    elif len(optimized_parameters) == 1:
        if isinstance(optimized_parameters[0], UniformIntegerHyperparameter):
            step = int((optimized_parameters[0].upper - optimized_parameters[0].lower) / n_test_samples)
            X_test = np.arange(optimized_parameters[0].lower, optimized_parameters[0].upper, step)
            if optimized_parameters[0].upper not in X_test:
                X_test = np.append(X_test, [optimized_parameters[0].upper])
            X_test = X_test.reshape(-1, 1)
        else:
            if optimized_parameters[0].log:
                X_test = np.logspace(
                    np.log(optimized_parameters[0].lower), np.log(optimized_parameters[0].upper), n_test_samples, base=np.e
                ).reshape(n_test_samples, 1)
            else:
                X_test = np.linspace(
                    optimized_parameters[0].lower, optimized_parameters[0].upper, n_test_samples
                ).reshape(n_test_samples, 1)
    else:
        X_test = None
        print("Not yet supported.")

    y_test = np.zeros((X_test.shape[1], X_test.shape[2]))
    param_dict = {}

    for n in range(X_test.shape[1]):
        for m in range(X_test.shape[2]):
            for i, param in enumerate(optimized_parameters):
                if isinstance(optimized_parameters[i], UniformIntegerHyperparameter):
                    param_dict[optimized_parameters[i].name] = int(X_test[i, n, m])
                else:
                    param_dict[optimized_parameters[i].name] = X_test[i, n, m]
            conf = Configuration(configuration_space=classifier.configspace, values=param_dict)
            y_test[n, m] = classifier.train(config=conf, seed=seed)
    # for conf in X_test_array.reshape(2, X_test_array.shape[1]*X_test_array.shape[2]).T:
    #     for i, param in enumerate(optimized_parameters):
    #         if isinstance(param, UniformIntegerHyperparameter):
    #             param_dict[param.name] = int(conf[i])
    #         else:
    # #             param_dict[param.name] = conf[i]
    #     X_test.append(Configuration(configuration_space=classifier.configspace, values=param_dict))
    # y_test = np.array([classifier.train(config=x, seed=seed) for x in X_test])
    # y_test = y_test.reshape(X_test_array.shape[1], X_test_array.shape[2])

    # log transform values of parameters that were log-sampled before training the symbolic models
    for i in range(len(optimized_parameters)):
        if optimized_parameters[i].log:
            X_train_smac[i, :] = np.log(X_train_smac[i, :])
            X_train_rand[i, :] = np.log(X_train_rand[i, :])
            X_test[i, :] = np.log(X_test[i, :])

    logger.info(
        f"Fit Symbolic Models for {model}."
    )

    symbolic_models = {}

    if symb_reg:
        # TODO: log symb regression logs?
        symb_params = dict(
            population_size=5000,
            generations=20,
            stopping_criteria=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            parsimony_coefficient=0.01,
            function_set=get_function_set(),
            metric="mean absolute error",
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

        # run SR on random samples
        symb_rand = SymbolicRegressor(**symb_params)
        symb_rand.fit(X_train_rand.T, y_train_rand)

        symbolic_models["Symb-smac"] = symb_smac
        symbolic_models["Symb-rand"] = symb_rand

        # write results to csv files
        df_scores_symb_reg = append_scores(
            df_scores_symb_reg,
            model,
            symb_smac,
            symb_rand,
            X_train_smac.T,
            y_train_smac,
            X_train_rand.T,
            y_train_rand,
            X_test.reshape(2, X_test.shape[1]*X_test.shape[2]).T,
            y_test.reshape(-1),
        )
        df_scores_symb_reg.to_csv(f"{res_dir}/scores_symb_reg.csv")

    df_expr[model] = {
        k: convert_symb(v, n_dim=len(optimized_parameters), n_decimals=3)
        for k, v in symbolic_models.items()
    }
    df_expr.to_csv(f"{res_dir}/functions.csv")

    # plot results
    if len(optimized_parameters) == 1:
        param_name = optimized_parameters[0].name
        param_log = optimized_parameters[0].log
        plot = plot_symb1d(
            X_train_smac=X_train_smac.T,
            y_train_smac=y_train_smac,
            X_train_rand=X_train_rand.T,
            y_train_rand=y_train_rand,
            X_test=X_test.T,
            y_test=y_test,
            xlabel=f"log({param_name})" if param_log else param_name,
            ylabel="Cost",
            symbolic_models=symbolic_models,
            function_name=model,
            xmin=optimized_parameters[0].lower,
            xmax=optimized_parameters[0].upper,
            plot_dir=plot_dir,
        )
    elif len(optimized_parameters) == 2:
        param0_name = optimized_parameters[0].name
        param0_log = optimized_parameters[0].log
        plot = plot_symb2d(
            X_train_smac=X_train_smac,
            X_train_rand=X_train_rand,
            X_test=X_test,
            y_test=y_test,
            function_name=model,
            symbolic_models=symbolic_models,
            parameters=optimized_parameters,
            plot_dir=plot_dir,
        )
