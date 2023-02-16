import logging
import dill as pickle

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor

from utils.utils import write_dict_to_cfg_file, get_hpo_test_data, get_scores, convert_symb
from utils.symb_reg_utils import get_function_set


if __name__ == "__main__":
    n_test_samples = 100
    symb_reg = True
    sampling_run_name = "smac_Quadratic function A_x_20230216_092250"

    # setup logging
    logger = logging.getLogger(__name__)

    run_dir = f"learning_curves/runs/{sampling_run_name}"
    sampling_dir = f"{run_dir}/sampling"
    model = sampling_run_name.split("_")[0]

    with open(f"{sampling_dir}/classifier.pkl", "rb") as classifier_file:
        classifier = pickle.load(classifier_file)

    optimized_parameters = classifier.configspace.get_hyperparameters()
    param_names = [param.name for param in optimized_parameters]

    X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, n_test_samples)

    df_train_samples = pd.read_csv(f"{sampling_dir}/samples.csv")
    seeds = df_train_samples.seed.unique()
    #n_samples_max = max(df_train_samples.groupby("seed")["cost"].count())

    n_samples_spacing = np.linspace(20, 100, 9)

    df_all_metrics = pd.DataFrame()
    df_all_expr = pd.DataFrame()

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
        metric="mse",  # "mean absolute error",
        random_state=0,
        verbose=1,
        const_range=(
            100,
            100,
        ),  # Range for constants, rather arbitrary setting here?
    )

    write_dict_to_cfg_file(
        dictionary=symb_params,
        target_file_path=f"{run_dir}/symbolic_regression_params.cfg",
    )

    for seed in seeds:
        X_train_all_samples = df_train_samples.query(f"seed == {seed}")[param_names]
        y_train_all_samples = df_train_samples.query(f"seed == {seed}")["cost"]

        for n_samples in n_samples_spacing.astype(int):

            X_train = X_train_all_samples[:n_samples]
            y_train = y_train_all_samples[:n_samples]

            logger.info(f"Fit Symbolic Model for {n_samples} samples and seed {seed}.")

            symbolic_models = {}

            if symb_reg:
                # run SR on SMAC samples
                symb_model = SymbolicRegressor(**symb_params)
                symb_model.fit(X_train, y_train)

                df_metrics = get_scores(X_train,
                                y_train,
                                X_test.reshape(len(optimized_parameters), -1).T,
                                y_test.reshape(-1),
                                symb_model)
                df_metrics.insert(0, "n_samples", n_samples)
                df_metrics.insert(0, "seed", seed)

                df_expr = pd.DataFrame(
                    {"expr": [convert_symb(symb_model, n_dim=len(optimized_parameters), n_decimals=3)]})
                df_expr.insert(0, "n_samples", n_samples)
                df_expr.insert(0, "seed", seed)

                df_all_metrics = pd.concat((df_all_metrics, df_metrics))
                df_all_expr = pd.concat((df_all_expr, df_expr))

    df_all_metrics.to_csv(f"{run_dir}/error_metrics.csv", index=False)
    df_all_expr.to_csv(f"{run_dir}/expressions.csv", index=False)
