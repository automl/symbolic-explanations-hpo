import os
import argparse
import numpy as np
import pandas as pd
import shutil
from sklearn.linear_model import LinearRegression

from utils.logging_utils import get_logger
from utils.run_utils import get_hpo_test_data, get_scores
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    parser.add_argument('--run_type',
                        choices=["smac", "rand", "surr"],
                        help=
                        '"smac": Linear regression is fitted on samples collected via Bayesian optimization, '
                        '"rand": Linear regression is fitted on randomly sampled configurations and their performance'
                        '"surr" Linear regression is fitted on random samples and their performance estimated '
                        'using the Gaussian process'
                        )
    args = parser.parse_args()
    run_type = args.run_type

    # number of HPs to optimize
    n_optimized_params = 2
    # number of HP combinations to consider per model
    max_hp_comb = 1

    n_samples_spacing = np.linspace(200, 200, 1, dtype=int).tolist()
    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8

    sampling_dir_name = "runs_sampling_hpobench"
    dir_with_test_data = "learning_curves/runs_surr_hpobench"
    n_test_samples = 100

    run_conf = get_run_config(job_id=args.job_id, n_optimized_params=n_optimized_params,
                              max_hp_comb=max_hp_comb)

    lin_dir_name = f"linreg"

    task_dict = get_task_dict()
    data_set_postfix = f"_{task_dict[run_conf['task_id']]}"
    optimized_parameters = list(run_conf["hp_conf"])
    model_name = get_benchmark_dict()[run_conf["benchmark"]]
    b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

    # add only parameters to be optimized to configspace
    cs = b.get_configuration_space(seed=0, hyperparameters=optimized_parameters)

    run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}{data_set_postfix}"

    sampling_dir = f"learning_curves/{sampling_dir_name}/{run_type}"
    sampling_run_dir = f"{sampling_dir}/{run_name}"

    lin_dir = f"results/runs_linreg_hpobench/{lin_dir_name}/{run_type}/{run_name}"
    if os.path.exists(lin_dir):
        shutil.rmtree(lin_dir)
    os.makedirs(f"{lin_dir}/lin_models")

    logger = get_logger(filename=f"{lin_dir}/lin_log.log")

    logger.info(f"Fit Linear Regression Model for {run_name} ({run_type}).")

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
    pd.DataFrame(X_test_reshaped, columns=optimized_parameters).to_csv(f"{lin_dir}/x_test.csv", index=False)
    pd.DataFrame(y_test_reshaped).to_csv(f"{lin_dir}/y_test.csv", header=False, index=False)

    df_all_metrics = pd.DataFrame()
    df_all_complexity = pd.DataFrame()
    df_all_expr = pd.DataFrame()

    for n_samples in n_samples_spacing:
        # For smac, get specific sampling file for each sample size for which the number of initial designs differs from
        # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
        if run_type == "surr" or (run_type == "smac" and init_design_max_ratio * n_samples < len(
                optimized_parameters) * init_design_n_configs_per_hyperparamter):
            df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{n_samples}.csv")
        else:
            df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{max(n_samples_spacing)}.csv")

        sampling_seeds = df_train_samples.seed.unique()

        for sampling_seed in sampling_seeds:
            X_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")[optimized_parameters]
            y_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")["cost"]


            if run_type == "surr":
                X_train = X_train_all_samples
                y_train = y_train_all_samples
            else:
                X_train = X_train_all_samples[:n_samples]
                y_train = y_train_all_samples[:n_samples]

            if len(X_train) < n_samples:
                logger.warning(
                    f"Found less than {n_samples} when trying to evaluate {n_samples} samples for sampling seed "
                    f"{sampling_seed}, skip.")
                break

            logger.info(f"Fit Linear Regression for {n_samples} samples and sampling seed {sampling_seed}.")

            # run SR on SMAC samples
            lin_model = LinearRegression()
            lin_model.fit(X_train, y_train)

            df_metrics = get_scores(
                y_train,
                lin_model.predict(X_train),
                y_test_reshaped,
                lin_model.predict(X_test_reshaped)
            )
            df_metrics.insert(0, "n_samples", n_samples)
            df_metrics.insert(1, "sampling_seed", sampling_seed)
            df_all_metrics = pd.concat((df_all_metrics, df_metrics))

            df_all_metrics.to_csv(f"{lin_dir}/error_metrics.csv", index=False)
