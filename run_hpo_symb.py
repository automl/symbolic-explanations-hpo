import logging
from os import path
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from smac import BlackBoxFacade
from smac.runhistory.encoder.encoder import convert_configurations_to_array
from ConfigSpace import (
    Configuration,
    UniformIntegerHyperparameter,
)

from symbolic_meta_model_wrapper import (
    SymbolicMetaModelWrapper,
    SymbolicPursuitModelWrapper,
)
from utils.symb_reg_utils import get_function_set
from utils.model_utils import get_classifier
from utils.utils import (
    get_output_dirs,
    convert_symb,
    append_scores,
    get_hpo_test_data,
    plot_symb1d,
    plot_symb2d,
    plot_symb2d_surrogate,
    write_dict_to_cfg_file,
)
from utils.smac_utils import run_smac_optimization

if __name__ == "__main__":
    seed = 3
    n_smac_samples = 10
    n_test_samples = 100
    model = "MLP"
    symb_reg = True
    symb_meta = False
    symb_purs = False

    train_on_surrogate = False
    compare_on_test = False

    classifier = get_classifier(model_name=model, seed=seed)

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
        facade=BlackBoxFacade,  # HyperparameterOptimizationFacade,
        target_function=classifier.train,
        function_name=model,
        n_eval=n_smac_samples,
        run_dir=run_dir,
        seed=seed,
    )
    X_train_smac = X_train_smac.astype(float)

    logger.info(f"Create grid configs for testing and train {model}.")

    # get test samples for SR
    X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, n_test_samples)

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

        if train_on_surrogate:
            # Accessing internal SMAC methods for this, is there a better way here?
            y_test_surrogate = np.zeros((X_test.shape[1], X_test.shape[2]))
            for i in range(X_test.shape[1]):
                for j in range(X_test.shape[2]):
                    x0 = (
                        int(X_test[0, i, j])
                        if isinstance(
                            optimized_parameters[0], UniformIntegerHyperparameter
                        )
                        else X_test[0, i, j]
                    )
                    x1 = (
                        int(X_test[1, i, j])
                        if isinstance(
                            optimized_parameters[1], UniformIntegerHyperparameter
                        )
                        else X_test[1, i, j]
                    )
                    conf = Configuration(
                        configuration_space=classifier.configspace,
                        values={
                            optimized_parameters[0].name: x0,
                            optimized_parameters[1].name: x1,
                        },
                    )
                    y_test_surrogate[i, j] = smac_facade._model.predict(
                        convert_configurations_to_array([conf])
                    )[0][0][0]
            y_test_surrogate = y_test_surrogate.reshape(-1)

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
            generations=3,
            # stopping_criteria=0.001,
            # p_crossover=0.7,
            # p_subtree_mutation=0.1,
            # p_hoist_mutation=0.05,
            # p_point_mutation=0.1,
            # max_samples=0.9,
            # parsimony_coefficient=0.01,
            function_set=get_function_set(),
            metric="mse",  # "mean absolute error",
            random_state=3,
            verbose=1,
            # const_range=(
            #     100,
            #     100,
            # ),  # Range for constants, rather arbitrary setting here?
        )

        write_dict_to_cfg_file(
            dictionary=symb_params,
            target_file_path=path.join(run_dir, "symbolic_regression_params.cfg"),
        )

        # run SR on SMAC samples
        symb_smac = SymbolicRegressor(**symb_params)
        if train_on_surrogate:
            symb_smac.fit(
                X_test.T.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0]),
                y_test_surrogate,
            )
        else:
            symb_smac.fit(X_train_smac.T, y_train_smac)
        symbolic_models["Symb-smac"] = symb_smac
        import dill as pickle
        with open(
                f"{res_dir}/symb.pkl", "wb") as symb_model_file:
            # pickling all programs lead to huge files
            delattr(symb_smac, "_programs")
            pickle.dump(symb_smac, symb_model_file)

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
        if train_on_surrogate:
            plot = plot_symb2d_surrogate(
                X_train_smac=X_train_smac,
                y_test_surrogate=y_test_surrogate,
                X_test=X_test,
                y_test=y_test,
                symbolic_models=symbolic_models,
                parameters=optimized_parameters,
                function_name=model,
                plot_dir=plot_dir,
            )
        else:
            plot = plot_symb2d(
                X_train_smac=X_train_smac,
                X_train_compare=X_train_compare,
                X_test=X_test,
                y_test=y_test,
                function_name=model,
                metric_name="1 - Accuracy",
                symbolic_models=symbolic_models,
                parameters=optimized_parameters,
                plot_dir=plot_dir,
            )
