import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.logging_utils import get_logger
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict

if __name__ == "__main__":
    include_surr_diff = False
    symb_dir_name = "parsimony0.0001"
    dir_with_test_data = "learning_curves/runs_surr_hpobench"
    n_optimized_params = 2
    max_hp_comb = 1

    labelsize = 16
    titlesize=18

    run_configs = get_run_config(n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)

    # Set up plot directories
    plot_dir = f"learning_curves/plots/combined_plots_hpobench_subplots_{symb_dir_name}"
    complexity_plot_dir = f"{plot_dir}/complexity"
    mse_plot_dir = f"{plot_dir}/mse"
    rmse_plot_dir = f"{plot_dir}/rmse"
    kt_plot_dir = f"{plot_dir}/kt"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(complexity_plot_dir)
    os.makedirs(mse_plot_dir)
    os.makedirs(rmse_plot_dir)
    os.makedirs(kt_plot_dir)

    logger = get_logger(filename=f"{plot_dir}/plot_log.log")

    logger.info(f"Save plots to {plot_dir}.")

    fig = plt.figure(figsize=(20, 25))

    for i, run_conf in enumerate(run_configs[:18]):

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
            # Load test data
            logger.info(f"Get test data.")
            X_test = np.array(pd.read_csv(f"{dir_with_test_data}/{run_name}/x_test.csv"))
            y_test = np.array(pd.read_csv(f"{dir_with_test_data}/{run_name}/y_test.csv"))

            avg_cost = y_test.mean()
            std_cost = y_test.std()

            df_error_metrics_all = pd.DataFrame()
            df_complexity_all = pd.DataFrame()

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


            logger.info(f"Create plots.")

            if model_name == "LR":
                classifier_title = "Logistic Regression"
            elif model_name == "RF":
                classifier_title = "Random Forest"
            elif model_name == "NN":
                classifier_title = "Neural Network"
            else:
                classifier_title = model_name
            param0 = f"log({optimized_parameters[0]})" if cs.get_hyperparameters()[0].log else optimized_parameters[0]
            param1 = f"log({optimized_parameters[1]})" if cs.get_hyperparameters()[1].log else optimized_parameters[1]

            ind = i + 1

            # Plot RMSE
            ax = plt.subplot(6, 3, ind)
            sns.boxplot(data=df_error_metrics_all, x="n_samples", y="rmse_test", hue="Experiment",
                        dodge=0.4, showfliers=False)
            plt.title(f"{classifier_title} ({param0}, {param1})\nDataset: {data_set}", fontsize=titlesize)
            plt.ylabel(f"RMSE $(c, s)$", fontsize=titlesize)
            plt.yticks(fontsize=labelsize)
            plt.xlabel("Number of Samples", fontsize=titlesize)
            plt.xticks(fontsize=labelsize)
            plt.legend([], [], frameon=False)

        except Exception as e:
            logger.warning(f"Could not process {run_name}: \n{e}")

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=titlesize)
    legend.get_title().set_fontsize(titlesize)
    plt.tight_layout()

    plt.savefig(f"{rmse_plot_dir}/boxplot.png", dpi=400)
    plt.close()