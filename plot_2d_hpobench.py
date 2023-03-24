import pandas as pd
import os
import numpy as np
import dill as pickle

from utils.run_utils import get_surrogate_predictions, get_hpo_test_data
from utils.plot_utils import plot_symb2d_subplots
from utils.logging_utils import get_logger

from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict


if __name__ == "__main__":
    # number of HPs to optimize
    n_optimized_params = 2
    # number of HP combinations to consider per model
    max_hp_comb = 1

    n_samples_spacing = np.linspace(20, 200, 10, dtype=int).tolist()
    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8

    dir_with_test_data = "" #"learning_curves/runs_surr_hpobench"
    n_test_samples = 100
    symb_dir_name = f"parsimony0.0001"
    n_samples = 200
    symb_seeds = [0]

    run_configs = get_run_config(n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)

    # set up directories
    plot_dir = f"results/plots"
    viz_plot_dir = f"{plot_dir}/visualization_hpobench"
    if not os.path.exists(viz_plot_dir):
        os.makedirs(viz_plot_dir)

    logger = get_logger(filename=f"{plot_dir}/plot_log.log")

    logger.info(f"Save plots to {plot_dir}.")

    for run_conf in run_configs:
        task_dict = get_task_dict()
        data_set = f"{task_dict[run_conf['task_id']]}"
        optimized_parameters = list(run_conf["hp_conf"])
        model_name = get_benchmark_dict()[run_conf["benchmark"]]
        b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

        # add only parameters to be optimized to configspace
        cs = b.get_configuration_space(hyperparameters=optimized_parameters)

        run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}_{data_set}"
        logger.info(f"Create plot for {run_name}.")

        sampling_dir_smac = f"results/runs_sampling_hpobench/smac/{run_name}"
        sampling_dir_rand = f"results/runs_sampling_hpobench/rand/{run_name}"

        symb_dir_surr = f"results/runs_symb_hpobench/{symb_dir_name}/surr/{run_name}/symb_models"
        symb_dir_smac = f"results/runs_symb_hpobench/{symb_dir_name}/smac/{run_name}/symb_models"
        symb_dir_rand = f"results/runs_symb_hpobench/{symb_dir_name}/rand/{run_name}/symb_models"

        # Get specific surrogate file for each sample size for which the number of initial designs differs from
        # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
        if init_design_max_ratio * n_samples < len(
                optimized_parameters) * init_design_n_configs_per_hyperparamter:
            n_eval = n_samples
        else:
            n_eval = max(n_samples_spacing)
        df_samples_smac = pd.read_csv(f"{sampling_dir_smac}/samples_{n_eval}.csv")
        df_samples_rand = pd.read_csv(f"{sampling_dir_rand}/samples_{max(n_samples_spacing)}.csv")

        # Load test data
        logger.info(f"Get test data for {run_name}.")
        if dir_with_test_data:
            X_test = get_hpo_test_data(b, cs.get_hyperparameters(), n_test_samples, return_x=True)
            y_test = np.array(
                pd.read_csv(f"{dir_with_test_data}/{run_name}/y_test.csv", header=None))
            y_test = y_test.reshape(X_test.shape[1], X_test.shape[2])
        else:
            logger.info(f"No previous test data dir provided, create test data for {run_name}.")
            X_test, y_test = get_hpo_test_data(b, cs.get_hyperparameters(), n_test_samples)

        for sampling_seed in df_samples_smac.seed.unique():
            logger.info(f"Considering sampling seed {sampling_seed}.")
            df_sampling_seed_smac = df_samples_smac.copy()[df_samples_smac["seed"] == sampling_seed]
            df_sampling_seed_rand = df_samples_rand.copy()[df_samples_rand["seed"] == sampling_seed]

            X_train_smac = np.array(df_sampling_seed_smac[[optimized_parameters[0], optimized_parameters[1]]])[:n_samples]
            X_train_rand = np.array(df_sampling_seed_rand[[optimized_parameters[0], optimized_parameters[1]]])[:n_samples]

            for symb_seed in symb_seeds:
                logger.info(f"Considering symb seed {symb_seed}.")

                predictions_test = {}

                with open(
                        f"{sampling_dir_smac}/surrogates/n_eval{n_eval}_samples{n_samples}_seed{sampling_seed}.pkl",
                        "rb") as surrogate_file:
                    surrogate_model = pickle.load(surrogate_file)
                predictions_test["GP Baseline"] = np.array(get_surrogate_predictions(
                    X_test.reshape(len(optimized_parameters), -1).T, cs, surrogate_model)).reshape(
                    X_test.shape[1], X_test.shape[2])
                with open(
                        f"{symb_dir_surr}/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl",
                        "rb") as symb_model_file_surr:
                    symb_surr = pickle.load(symb_model_file_surr)
                symb_pred_surr = symb_surr.predict(
                        X_test.T.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0])
                    ).reshape(X_test.shape[2], X_test.shape[1]).T
                predictions_test[f"SR (BO-GP)"] = symb_pred_surr
                surr_dir = f"results/runs_surr/{run_name}"
                with open(
                        f"{symb_dir_smac}/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl",
                        "rb") as symb_model_file_smac:
                    symb_smac = pickle.load(symb_model_file_smac)
                symb_pred_smac = symb_smac.predict(
                        X_test.T.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0])
                    ).reshape(X_test.shape[2], X_test.shape[1]).T
                predictions_test[f"SR (BO)"] = symb_pred_smac
                with open(
                        f"{symb_dir_rand}/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl",
                        "rb") as symb_model_file_rand:
                    symb_rand = pickle.load(symb_model_file_rand)
                symb_prad_rand = symb_rand.predict(
                        X_test.T.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0])
                    ).reshape(X_test.shape[2], X_test.shape[1]).T
                predictions_test[f"SR (Random)"] = symb_prad_rand

                X_train_list = [None, None, X_train_smac.T, X_train_rand.T]

                filename = f"{run_name}_n_samples{n_samples}_" \
                           f"sampling_seed{sampling_seed}_symb_seed{symb_seed}"

                plot_symb2d_subplots(
                                X_train_list=X_train_list,
                                X_test=X_test,
                                y_test=y_test,
                                function_name=model_name,
                                metric_name=r'Validation Loss',
                                predictions_test=predictions_test,
                                parameters=cs.get_hyperparameters(),
                                plot_dir=viz_plot_dir,
                                filename=filename
                            )