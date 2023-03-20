import pandas as pd
import os
import shutil
import numpy as np

from utils.logging_utils import get_logger
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict

if __name__ == "__main__":
    symb_dir_name = "parsimony0005"
    dir_with_test_data = "learning_curves/runs_surr_hpobench"
    n_optimized_params = 2

    run_configs = get_run_config(n_optimized_params=n_optimized_params)

    # Set up plot directories
    metric_dir = f"learning_curves/metrics/{symb_dir_name}"
    if os.path.exists(metric_dir):
        shutil.rmtree(metric_dir)
    os.makedirs(metric_dir)

    logger = get_logger(filename=f"{metric_dir}/metric_log.log")

    logger.info(f"Save plots to {metric_dir}.")

    for run_conf in run_configs:

        task_dict = get_task_dict()
        data_set = f"{task_dict[run_conf['task_id']]}"
        optimized_parameters = list(run_conf["hp_conf"])
        model_name = get_benchmark_dict()[run_conf["benchmark"]]
        b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

        # add only parameters to be optimized to configspace
        cs = b.get_configuration_space(hyperparameters=optimized_parameters)

        run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}_{data_set}"

        try:
            # Load test data
            logger.info(f"Get test data.")
            X_test = np.array(pd.read_csv(f"{dir_with_test_data}/{run_name}/x_test.csv"))
            y_test = np.array(pd.read_csv(f"{dir_with_test_data}/{run_name}/y_test.csv"))

            avg_cost = y_test.mean()
            std_cost = y_test.std()

            run_rmse_mean, run_rmse_std = {}, {}

            for sampling_type in ["SR (BO)", "SR (Random)", "SR (BO-GP)", "GP (BO)"]:

                if sampling_type == "GP (BO)":
                    symb_dir = f"learning_curves/runs_surr_hpobench/{run_name}"
                else:
                    if sampling_type == "SR (BO)":
                        symb_dir = f"learning_curves/runs_symb_hpobench/{symb_dir_name}/smac/{run_name}"
                    elif sampling_type == "SR (Random)":
                        symb_dir = f"learning_curves/runs_symb_hpobench/{symb_dir_name}/rand/{run_name}"
                    else:
                        symb_dir = f"learning_curves/runs_symb_hpobench/{symb_dir_name}/surr/{run_name}"

                df_error_metrics = pd.read_csv(f"{symb_dir}/error_metrics.csv")
                df_error_metrics["rmse_test"] = np.sqrt(df_error_metrics["mse_test"])
                df_error_metrics["rmse_train"] = np.sqrt(df_error_metrics["mse_train"])
                run_rmse_mean[sampling_type] = df_error_metrics["rmse_test"].mean(axis=0)

            run_rmse_mean["Test Mean"] = avg_cost
            run_rmse_std["Test Std"] = std_cost
            df_run_rmse_mean = pd.DataFrame(run_rmse_mean, index=[run_name])
            df_run_rmse_std = pd.DataFrame(run_rmse_std, index=[run_name])
            df_run_rmse_mean.to_csv(f"{metric_dir}/rmse_mean.csv")
            df_run_rmse_mean.to_csv(f"{metric_dir}/rmse_std.csv")

        except Exception as e:
            logger.warning(f"Could not process {run_name}: \n{e}")
