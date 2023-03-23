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

    labelsize = 12
    titlesize=13

    run_configs = get_run_config(n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)

    # Set up plot directories
    plot_dir = f"learning_curves/plots/combined_plots_hpobench_{symb_dir_name}"
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

    fig = plt.figure(figsize=(15, 8))

    for i, run_conf in enumerate(run_configs):

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

                if include_surr_diff:
                    df_error_metrics = pd.read_csv(
                        f"learning_curves/runs_symb_hpobench/{symb_dir_name}/surr/{run_name}/error_metrics_compare_surr.csv")
                    df_error_metrics["rmse_test"] = np.sqrt(df_error_metrics["mse_test"])
                    df_error_metrics["rmse_train"] = np.sqrt(df_error_metrics["mse_train"])
                    df_error_metrics.insert(0, "Experiment", f"RMSE $(c, s)$")
                    df_error_metrics_all = pd.concat((df_error_metrics_all, df_error_metrics))

            logger.info(f"Create plots.")

            classifier_title = model_name

            param0 = f"log({optimized_parameters[0]})" if cs.get_hyperparameters()[0].log else optimized_parameters[0]
            param1 = f"log({optimized_parameters[1]})" if cs.get_hyperparameters()[1].log else optimized_parameters[1]

            if i == 1:
                ind = 3
            elif i == 2:
                ind = 2
            else:
                ind = i + 1

            # Plot RMSE
            ax = plt.subplot(2, 2, ind)
            #line = plt.axhline(y=std_cost, color='darkred', linestyle='--', linewidth=0.5, label="Std.")
            sns.boxplot(data=df_error_metrics_all, x="n_samples", y="rmse_test", hue="Experiment",
                        dodge=0.4, showfliers=False)
            #sns.pointplot(data=df_error_metrics_all, x="n_samples", y="rmse_test", hue="Experiment", errorbar="sd",
            #              linestyles="", capsize=0.2, errwidth=0.7, scale=0.7, dodge=0.4)#, showfliers=False)
            plt.title(f"Dataset: {data_set}\n{classifier_title} ({param0}, {param1})", fontsize=titlesize)
            #plt.title(f"Test Mean: {avg_cost:.3f}, Test Std.: {std_cost:.3f}", fontsize=10),
            plt.ylabel(f"RMSE $(c, s)$", fontsize=titlesize)
            plt.yticks(fontsize=labelsize)
            if ind == 3 or i == 4:
                plt.xlabel("Number of Samples", fontsize=titlesize)
            else:
                plt.xlabel("")
            plt.xticks(fontsize=labelsize)
            #plt.ylim(0., 0.4)
            #plt.tight_layout(rect=(0, 0.05, 1, 1))
            plt.legend([], [], frameon=False)
            if ind == 1:
                plt.figtext(0.5, 0.98, f"Dataset: {data_set}", ha="center", va="top", fontsize=titlesize)
            if ind == 3:
                plt.figtext(0.5, 0.50, f"Dataset: {data_set}", ha="center", va="top", fontsize=titlesize)

                # # Plot Kendall
                # plt.figure()
                # _, ax = plt.subplots(figsize=(8, 5))
                # sns.pointplot(data=df_error_metrics_all, x="n_samples", y="kt_test", hue="Experiment", errorbar="sd",
                #               linestyles="", capsize=0.2, errwidth=0.7, scale=0.7, dodge=0.4)#, showfliers=False)
                # if data_set:
                #     plt.title(f"{classifier_title}, Dataset: {data_set}\nOptimize: {param0}, {param1}", fontsize=titlesize)
                # else:
                #     plt.title(f"{classifier_title}\nOptimize: {param0}, {param1}", fontsize=titlesize)
                # plt.ylabel("Test Kendall Tau", fontsize=titlesize)
                # plt.yticks(fontsize=labelsize)
                # plt.xlabel("Number of Samples", fontsize=titlesize)
                # plt.xticks(fontsize=labelsize)
                # plt.ylim(-0.6, 1.)
                # plt.tight_layout(rect=(0, 0.05, 1, 1))
                # sns.move_legend(
                #     ax, "lower center",
                #     bbox_to_anchor=(0.45, -0.27),
                #     ncol=5,
                #     title=None, frameon=False,
                #     fontsize=labelsize
                # )
                # plt.savefig(f"{kt_plot_dir}/{run_name}_pointplot.png", dpi=400)
                # plt.close()
                # 
                # # Plot Complexity
                # plt.figure()
                # _, ax = plt.subplots(figsize=(8, 5))
                # sns.pointplot(data=df_complexity_all, x="n_samples", y="program_operations", hue="Experiment", errorbar="sd",
                #               linestyles="", capsize=0.2, errwidth=0.7, scale=0.7, dodge=0.2)#, showfliers=False)
                # if data_set:
                #     plt.title(f"{classifier_title}, Dataset: {data_set}\nOptimize: {param0}, {param1}", fontsize=titlesize)
                # else:
                #     plt.title(f"{classifier_title}\nOptimize: {param0}, {param1}", fontsize=titlesize)
                # plt.ylabel("Number of Operations in Formula", fontsize=titlesize)
                # plt.xlabel("Number of Samples", fontsize=titlesize)
                # plt.xticks(fontsize=labelsize)
                # #plt.ylim(0, 18.5)
                # #plt.yticks(np.arange(0, 20, 2.0), fontsize=labelsize)
                # plt.tight_layout(rect=(0, 0.05, 1, 1))
                # sns.move_legend(
                #     ax, "lower center",
                #     bbox_to_anchor=(0.45, -0.27),
                #     ncol=3,
                #     title=None, frameon=False,
                #     fontsize=labelsize
                # )
                # plt.savefig(f"{complexity_plot_dir}/{run_name}_complexity_pointplot.png", dpi=400)
                # plt.close()

        except Exception as e:
            logger.warning(f"Could not process {run_name}: \n{e}")

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, ncol=4, loc='lower center', frameon=False, fontsize=titlesize)
    legend.get_title().set_fontsize(titlesize)
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    
    plt.savefig(f"{rmse_plot_dir}/boxplot.png", dpi=400)
    plt.close()