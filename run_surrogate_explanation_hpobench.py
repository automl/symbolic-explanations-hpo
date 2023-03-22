import os
import shutil
import dill as pickle
import argparse
import numpy as np
import pandas as pd

from utils.logging_utils import get_logger
from utils.run_utils import get_hpo_test_data, get_scores, get_surrogate_predictions
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    args = parser.parse_args()

    max_hp_comb = None

    sampling_dir_name = "runs_sampling_hpobench"
    # only for loading test data
    dir_with_test_data = ""
    n_optimized_params = 2
    n_samples_spacing = np.linspace(20, 200, 10, dtype=int).tolist()
    n_test_samples = 100
    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8

    run_conf = get_run_config(job_id=args.job_id, n_optimized_params=n_optimized_params,
                              max_hp_comb=max_hp_comb)

    parsimony_coefficient = run_conf["parsimony"]
    symb_dir_name = f"parsimony{parsimony_coefficient}"

    task_dict = get_task_dict()
    data_set_postfix = f"_{task_dict[run_conf['task_id']]}"
    optimized_parameters = list(run_conf["hp_conf"])
    model_name = get_benchmark_dict()[run_conf["benchmark"]]
    b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

    # add only parameters to be optimized to configspace
    cs = b.get_configuration_space(seed=0, hyperparameters=optimized_parameters)

    run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}{data_set_postfix}"

    sampling_dir = f"learning_curves/{sampling_dir_name}/smac"
    sampling_run_dir = f"{sampling_dir}/{run_name}"
    surr_dir = f"learning_curves/runs_surr_hpobench/{run_name}"
    if os.path.exists(surr_dir):
        shutil.rmtree(surr_dir)
    os.makedirs(surr_dir)
    os.makedirs(f"{surr_dir}/surr_preds")

    # setup logging
    logger = get_logger(filename=f"{surr_dir}/surrogate_log.log")

    logger.info(f"Evaluate surrogate model for {run_name}.")

    # Load test data
    logger.info(f"Get and save test data.")
    if dir_with_test_data:
        X_test = get_hpo_test_data(b, cs.get_hyperparameters(), n_test_samples, return_x=True)
        y_test = np.array(
        pd.read_csv(f"{dir_with_test_data}/{run_name}/y_test.csv", header=None))
        y_test = y_test.reshape(X_test.shape[1], X_test.shape[2])
    else:
        logger.info(f"No previous test data dir provided, create test data for {run_name}.")
        X_test, y_test = get_hpo_test_data(b, cs.get_hyperparameters(), n_test_samples)
    X_test_reshaped = X_test.reshape(len(optimized_parameters), -1).T
    y_test_reshaped = y_test.reshape(-1)
    pd.DataFrame(X_test_reshaped, columns=optimized_parameters).to_csv(f"{surr_dir}/x_test.csv", index=False)
    pd.DataFrame(y_test_reshaped).to_csv(f"{surr_dir}/y_test.csv", header=False, index=False)

    df_all_metrics = pd.DataFrame()

    for n_samples in n_samples_spacing:
        # Get specific surrogate file for each sample size for which the number of initial designs differs from
        # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
        if init_design_max_ratio * n_samples < len(
                optimized_parameters) * init_design_n_configs_per_hyperparamter:
            n_eval = n_samples
        else:
            n_eval = max(n_samples_spacing)

        df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{n_eval}.csv")

        sampling_seeds = df_train_samples.seed.unique()

        for sampling_seed in sampling_seeds:
            X_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")[optimized_parameters]
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
                with open(f"{sampling_run_dir}/surrogates/n_eval{n_eval}_samples{n_samples}_seed{sampling_seed}.pkl",
                          "rb") as surrogate_file:
                    surrogate_model = pickle.load(surrogate_file)

                df_metrics = get_scores(
                    y_train,
                    get_surrogate_predictions(np.array(X_train), cs, surrogate_model),
                    y_test.reshape(-1),
                    get_surrogate_predictions(X_test.reshape(len(optimized_parameters), -1).T, cs, surrogate_model),
                )

                df_metrics.insert(0, "n_samples", n_samples)
                df_metrics.insert(0, "sampling_seed", sampling_seed)
                df_all_metrics = pd.concat((df_all_metrics, df_metrics))

                # Save surrogate test predictions
                test_pred = get_surrogate_predictions(X_test.reshape(len(optimized_parameters), -1).T, cs, surrogate_model)
                pd.DataFrame(test_pred).to_csv(f"{surr_dir}/surr_preds/test_pred_samples{n_samples}_seed{sampling_seed}.csv", index=False)
            except:
                logger.warning(f"File n_eval{n_eval}_samples{n_samples}_seed{sampling_seed}.pkl could not be loaded, "
                               f"skip.")

            df_all_metrics.to_csv(f"{surr_dir}/error_metrics.csv", index=False)
