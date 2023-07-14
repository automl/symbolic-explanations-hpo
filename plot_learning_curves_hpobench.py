import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

from utils.logging_utils import get_logger
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    args = parser.parse_args()

    # number of HPs to optimize
    n_optimized_params = 2
    # number of HP combinations to consider per model
    max_hp_comb = 1

    symb_dir_name = "parsimony0.0001"
    labelsize = 12
    titlesize=13

    if args.job_id:
        run_configs = [
            get_run_config(job_id=args.job_id, n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)]
    else:
        run_configs = get_run_config(n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)

    # Set up plot directories
    plot_dir = f"results/plots/combined_plots_hpobench_{symb_dir_name}"
    rmse_plot_dir = f"{plot_dir}/rmse"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(rmse_plot_dir)

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

        try:
            df_error_metrics_all = pd.DataFrame()
            df_complexity_all = pd.DataFrame()

            for sampling_type in ["SR (BO)", "SR (Random)", "SR (BO-GP)", "GP Baseline"]:

                try:
                    if sampling_type == "GP Baseline":
                        symb_dir = f"results/runs_surr_hpobench/{run_name}"
                    else:
                        if sampling_type == "SR (BO)":
                            symb_dir = f"results/runs_symb_hpobench/{symb_dir_name}/smac/{run_name}"
                        elif sampling_type == "SR (Random)":
                            symb_dir = f"results/runs_symb_hpobench/{symb_dir_name}/rand/{run_name}"
                        else:
                            symb_dir = f"results/runs_symb_hpobench/{symb_dir_name}/surr/{run_name}"
                        df_complexity = pd.read_csv(f"{symb_dir}/complexity.csv")
                        df_complexity.insert(0, "Experiment", f"{sampling_type}")
                        df_complexity_all = pd.concat((df_complexity_all, df_complexity))
                        df_complexity_all = df_complexity_all[df_complexity_all["program_operations"] != -1]

                    df_error_metrics = pd.read_csv(f"{symb_dir}/error_metrics.csv")
                    df_error_metrics["rmse_test"] = np.sqrt(df_error_metrics["mse_test"])
                    df_error_metrics["rmse_train"] = np.sqrt(df_error_metrics["mse_train"])
                    df_error_metrics.insert(0, "Experiment", f"{sampling_type}")
                    df_error_metrics_all = pd.concat((df_error_metrics_all, df_error_metrics))
                except Exception as e:
                    logger.warning(f"Could not process {sampling_type} for {run_name}: \n{e}")

                classifier_title = model_name

                param0 = optimized_parameters[0]
                param1 = optimized_parameters[1]

                # Plot RMSE
                plt.figure()
                _, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df_error_metrics_all, x="n_samples", y="rmse_test", hue="Experiment",
                            dodge=0.4, showfliers=False)
                plt.title(f"Dataset: {data_set}\n{classifier_title} ({param0}, {param1})", fontsize=titlesize)
                plt.ylabel(f"RMSE $(c, s)$", fontsize=titlesize)
                plt.yticks(fontsize=labelsize)
                plt.xlabel("Number of Samples", fontsize=titlesize)
                plt.xticks(fontsize=labelsize)
                plt.tight_layout(rect=(0, 0.05, 1, 1))
                sns.move_legend(
                    ax, "lower center",
                    bbox_to_anchor=(0.45, -0.27),
                    ncol=4,
                    title=None, frameon=False,
                    fontsize=labelsize
                )
                plt.savefig(f"{rmse_plot_dir}/{run_name}_pointplot.png", dpi=400)
                plt.close()

        except Exception as e:
            logger.warning(f"Could not process {run_name}: \n{e}")

