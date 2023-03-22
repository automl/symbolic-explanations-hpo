import pandas as pd
import os
import shutil
import numpy as np

from utils.logging_utils import get_logger
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict
from utils.latex_utils import generate_result_table

if __name__ == "__main__":
    # parsimony_coefficient_space = [
    #     0.000001, 0.0000025, 0.000005, 0.0000075,
    #     0.00001, 0.000025, 0.00005, 0.000075,
    #     0.0001, 0.00025, 0.00075, 0.0005,
    #     0.001, 0.0025, 0.005, 0.0075,
    #     0.01, 0.025, 0.05, 0.075
    # ]
    parsimony_coefficient_space = [0.0001]

    dir_with_test_data = "learning_curves/runs_surr_hpobench"
    n_optimized_params = 2
    # if None, average metrics over all sample sizes
    eval_at_n_samples = 200
    max_hp_comb = 1
    to_latex = True

    run_configs = get_run_config(n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)

    for parsimony in parsimony_coefficient_space:
        symb_dir_name = f"parsimony{parsimony}"

        # Set up plot directories
        metric_dir = f"learning_curves/metrics/{symb_dir_name}"
        if os.path.exists(metric_dir):
            shutil.rmtree(metric_dir)
        os.makedirs(metric_dir)
    
        logger = get_logger(filename=f"{metric_dir}/metric_log.log")
    
        logger.info(f"Save metrics to {metric_dir}.")
    
        df_run_rmse_mean_all = pd.DataFrame()
        df_run_rmse_std_all = pd.DataFrame()
        df_run_count_all = pd.DataFrame()
    
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
    
                run_count, run_rmse_mean, run_rmse_std = {}, {}, {}
    
                for sampling_type in ["SR (BO)", "SR (Random)", "SR (BO-GP)", "GP Baseline"]:
                    try:
                        if sampling_type == "GP Baseline":
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
    
                        if eval_at_n_samples:
                            df_error_metrics = df_error_metrics[df_error_metrics["n_samples"] == eval_at_n_samples]
    
                        run_count[sampling_type] = df_error_metrics["rmse_test"].count()
                        run_rmse_mean[sampling_type] = df_error_metrics["rmse_test"].mean(axis=0)
                        run_rmse_std[sampling_type] = df_error_metrics["rmse_test"].std(axis=0)
                    except Exception as e:
                        logger.warning(f"Could not process {sampling_type} for {run_name}: \n{e}")
    
                if eval_at_n_samples:
                    n_samples_postfix = f"_n_samples{eval_at_n_samples}"
                else:
                    n_samples_postfix = "_all_sample_sizes"
    
                df_run_count = pd.DataFrame(run_count, index=[f"{model_name}:({', '.join(optimized_parameters)}):{data_set}"])
                df_run_count_all = pd.concat((df_run_count_all, df_run_count))
                df_run_count_all.to_csv(f"{metric_dir}/count{n_samples_postfix}.csv")
    
                #run_rmse_mean["Test Mean"] = avg_cost
                df_run_rmse_mean = pd.DataFrame(run_rmse_mean, index=[f"{model_name} ({', '.join(optimized_parameters)}):{data_set}"])
                df_run_rmse_mean.insert(0, "Model", model_name)
                df_run_rmse_mean.insert(1, "Dataset", data_set)
                df_run_rmse_mean_all = pd.concat((df_run_rmse_mean_all, df_run_rmse_mean))
                df_run_rmse_mean_all.to_csv(f"{metric_dir}/rmse_mean{n_samples_postfix}.csv")
    
                #run_rmse_std["Test Std"] = std_cost
                df_run_rmse_std = pd.DataFrame(run_rmse_std, index=[f"{model_name} ({', '.join(optimized_parameters)}):{data_set}"])
                df_run_rmse_std_all = pd.concat((df_run_rmse_std_all, df_run_rmse_std))
                df_run_rmse_std_all.to_csv(f"{metric_dir}/rmse_std{n_samples_postfix}.csv")

            except Exception as e:
                logger.warning(f"Could not process {run_name}: \n{e}")

        if to_latex:
            latex_out = generate_result_table(df_run_rmse_mean_all, df_run_rmse_std_all, stddev=True,
                                              show_avg_and_median=False)

            with open(f"{metric_dir}/rmse_latex.txt", "w") as f:
                f.write(latex_out)
