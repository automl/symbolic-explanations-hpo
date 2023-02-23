import os
import logging
import dill as pickle
import argparse
import sys
import numpy as np
import pandas as pd

from utils.utils import get_hpo_test_data, get_scores, get_surrogate_predictions
from utils import functions

sys.modules['functions'] = functions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    args = parser.parse_args()
    job_id = args.job_id

    n_test_samples = 100

    sampling_run_names = [
        "smac_BDT_learning_rate_n_estimators_digits_20230223_162320",
        "smac_BDT_learning_rate_n_estimators_iris_20230223_162320",
        "smac_Branin_2D_X0_X1_20230223_162155",
        "smac_Camelback_2D_X0_X1_20230223_162155",
        "smac_Exponential_function_2D_X0_X1_20230223_162156",
        "smac_Linear_2D_X0_X1_20230223_162155",
        "smac_MLP_learning_rate_init_max_iter_digits_20230223_162437",
        "smac_MLP_learning_rate_init_max_iter_iris_20230223_162436",
        "smac_MLP_learning_rate_init_n_layer_digits_20230223_162436",
        "smac_MLP_learning_rate_init_n_layer_iris_20230223_162436",
        "smac_MLP_learning_rate_init_n_neurons_digits_20230223_162436",
        "smac_MLP_learning_rate_init_n_neurons_iris_20230223_162436",
        "smac_MLP_max_iter_n_layer_digits_20230223_162436",
        "smac_MLP_max_iter_n_layer_iris_20230223_162436",
        "smac_MLP_max_iter_n_neurons_digits_20230223_162436",
        "smac_MLP_max_iter_n_neurons_iris_20230223_162436",
        "smac_MLP_n_layer_n_neurons_digits_20230223_162436",
        "smac_MLP_n_layer_n_neurons_iris_20230223_162437",
        "smac_Polynom_function_2D_X0_X1_20230223_162156",
        "smac_Rosenbrock_2D_X0_X1_20230223_162155",
        "smac_SVM_C_coef0_digits_20230223_162857",
        "smac_SVM_C_coef0_digits_20230223_164415",
        "smac_SVM_C_coef0_iris_20230223_162859",
        "smac_SVM_C_degree_digits_20230223_162900",
        "smac_SVM_C_degree_iris_20230223_162900",
        "smac_SVM_C_degree_iris_20230223_164415",
        "smac_SVM_C_gamma_digits_20230223_162900",
        "smac_SVM_C_gamma_iris_20230223_162859",
        "smac_SVM_coef0_degree_digits_20230223_162859",
        "smac_SVM_coef0_degree_iris_20230223_162859",
        "smac_SVM_coef0_gamma_digits_20230223_162859",
        "smac_SVM_coef0_gamma_iris_20230223_162859",
        "smac_SVM_degree_gamma_digits_20230223_162900",
        "smac_SVM_degree_gamma_iris_20230223_162859"
    ]
    sampling_run_name = sampling_run_names[int(job_id)]

    run_dir = f"learning_curves/runs/{sampling_run_name}"
    sampling_dir = f"{run_dir}/sampling"
    if not os.path.exists(f"{run_dir}/symb_models"):
        os.makedirs(f"{run_dir}/symb_models")
    model = sampling_run_name.split("_")[1]

    # setup logging
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(filename=f"{run_dir}/surrogate_log.log", encoding="utf8")
    handler.setLevel("INFO")
    handler.setFormatter(
        logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    logger.root.addHandler(handler)
    handler2 = logging.StreamHandler()
    handler2.setLevel("INFO")
    handler2.setFormatter(
        logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    handler2.setStream(sys.stdout)
    logger.root.addHandler(handler2)
    logger.root.setLevel("INFO")

    with open(f"{sampling_dir}/classifier.pkl", "rb") as classifier_file:
        classifier = pickle.load(classifier_file)

    optimized_parameters = classifier.configspace.get_hyperparameters()
    param_names = [param.name for param in optimized_parameters]

    logger.info(f"Fit Symbolic Model for {sampling_run_name}.")

    X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, n_test_samples)

    df_train_samples = pd.read_csv(f"{sampling_dir}/samples.csv")
    sampling_seeds = df_train_samples.seed.unique()

    n_samples_spacing = np.linspace(20, 200, 10)

    df_all_metrics = pd.DataFrame()

    for sampling_seed in sampling_seeds:
        X_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")[param_names]
        y_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")["cost"]

        for n_samples in n_samples_spacing.astype(int):

            X_train = X_train_all_samples[:n_samples]
            y_train = y_train_all_samples[:n_samples]

            logger.info(f"Evaluate Surrogate Model for {n_samples} samples and sampling seed {sampling_seed}.")

            # load surrogate model
            with open(f"{run_dir}/sampling/surrogates/seed{sampling_seed}_samples{n_samples}.pkl",
                      "rb") as surrogate_file:
                surrogate_model = pickle.load(surrogate_file)

            df_metrics = get_scores(
                y_train,
                get_surrogate_predictions(np.array(X_train), classifier, surrogate_model),
                y_test.reshape(-1),
                get_surrogate_predictions(X_test.reshape(len(optimized_parameters), -1).T, classifier, surrogate_model),
            )

            df_metrics.insert(0, "n_samples", n_samples)
            df_metrics.insert(0, "sampling_seed", sampling_seed)
            df_all_metrics = pd.concat((df_all_metrics, df_metrics))

            df_all_metrics.to_csv(f"{run_dir}/surrogate_error_metrics.csv", index=False)
