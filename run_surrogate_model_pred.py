import os
import shutil
import dill as pickle
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

from utils.utils import get_hpo_test_data, get_scores, get_surrogate_predictions
from utils.functions_utils import get_functions2d, NamedFunction
from utils.model_utils import get_hyperparams, get_classifier_from_run_conf
from utils.logging_utils import get_logger


N_SAMPLES_SPACING = np.linspace(20, 200, 10, dtype=int).tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    args = parser.parse_args()
    job_id = args.job_id

    functions = get_functions2d()
    models = ["MLP", "SVM", "BDT", "DT"]
    #models = functions
    data_sets = ["digits", "iris"]

    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8
    sampling_dir_name = "runs_sampling"

    n_test_samples = 100

    run_configs = []
    for model in models:
        if isinstance(model, NamedFunction):
            run_configs.append({"model": model, "data_set_name": None})
        else:
            hyperparams = get_hyperparams(model_name=model)
            hp_comb = combinations(hyperparams, 2)
            for hp_conf in hp_comb:
                for ds in data_sets:
                    run_configs.append({"model": model, hp_conf[0]: True, hp_conf[1]: True, "data_set_name": ds})
    run_conf = run_configs[int(job_id)]
    if run_conf['data_set_name']:
        data_set_postfix = f"_{run_conf['data_set_name']}"
    else:
        data_set_postfix = ""
    model = run_conf.pop("model")
    if isinstance(model, NamedFunction):
        classifier = model
    else:
        classifier = get_classifier_from_run_conf(model_name=model, run_conf=run_conf)

    function_name = classifier.name if isinstance(classifier, NamedFunction) else model
    optimized_parameters = classifier.configspace.get_hyperparameters()
    parameter_names = [param.name for param in optimized_parameters]

    run_name = f"{function_name.replace(' ', '_')}_{'_'.join(parameter_names)}{data_set_postfix}"

    sampling_dir = f"learning_curves/{sampling_dir_name}/smac"
    sampling_run_dir = f"{sampling_dir}/{run_name}"
    surr_dir = f"learning_curves/runs_surr/{run_name}"
    if os.path.exists(surr_dir):
        shutil.rmtree(surr_dir)
    os.makedirs(surr_dir)

    # setup logging
    logger = get_logger(filename=f"{surr_dir}/surrogate_log.log")

    logger.info(f"Evaluate surrogate model for {run_name}.")

    # Load test data
    logger.info(f"Get test data.")
    try:
        X_test = np.array(
            pd.read_csv(f"learning_curves/runs_symb/default/smac/{run_name}/x_test.csv", header=False))
        y_test = np.array(pd.read_csv(f"learning_curves/runs_symb/default/smac/{run_name}/y_test.csv"))
    except:
        logger.info(f"No test data found, create test data for {run_name}.")
        X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, n_test_samples)

    df_all_metrics = pd.DataFrame()

    for n_samples in N_SAMPLES_SPACING:
        # Get specific surrogate file for each sample size for which the number of initial designs differs from
        # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
        if init_design_max_ratio * n_samples < len(
                optimized_parameters) * init_design_n_configs_per_hyperparamter:
            n_eval = n_samples
        else:
            n_eval = max(N_SAMPLES_SPACING)

        df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{n_eval}.csv")

        sampling_seeds = df_train_samples.seed.unique()

        for sampling_seed in sampling_seeds:
            X_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")[parameter_names]
            y_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")["cost"]

            X_train = X_train_all_samples[:n_samples]
            y_train = y_train_all_samples[:n_samples]

            if len(X_train) < n_samples:
                logger.warning(
                    f"Found less than {n_samples} when trying to evaluate {n_samples} samples for sampling seed "
                    f"{sampling_seed}, skip.")
                break

            logger.info(f"Evaluate Surrogate Model for {n_samples} samples and sampling seed {sampling_seed}.")

            # Load surrogate model
            try:
                with open(f"{sampling_dir}/surrogates/n_eval{n_eval}_samples{n_samples}_seed{sampling_seed}.pkl",
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
            except:
                logger.warning(f"File n_eval{n_eval}_samples{n_samples}_seed{sampling_seed}.pkl could not be loaded, "
                               f"skip.")

            df_all_metrics.to_csv(f"{surr_dir}/error_metrics.csv", index=False)
