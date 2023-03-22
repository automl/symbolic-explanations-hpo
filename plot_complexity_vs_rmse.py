import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.logging_utils import get_logger
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict

if __name__ == "__main__":
    parsimony_coefficient_space = [
        # 0.000001, 0.0000025, 0.000005, 0.0000075,
        # 0.00001, 0.000025, 0.00005,
        # 0.000075,
        0.0001, 0.00025, 0.0005, 0.00075,
        0.001, 0.0025, 0.005, 0.0075,
        0.01, 0.025, 0.05, 0.075
    ]
    n_optimized_params = 2
    n_samples = 200

    labelsize = 14
    titlesize=16

    run_configs = get_run_config(n_optimized_params=n_optimized_params, max_hp_comb=1)

    # Set up plot directories
    plot_dir = f"learning_curves/plots/complexity_vs_rmse_hpobench"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

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

        logger.info(f"############################# Create plot for {run_name}.")

        df_joined_all = pd.DataFrame()

        for parsimony in parsimony_coefficient_space:

            logger.info(f"########## Evaluate parsimony {parsimony}")

            symb_dir = f"learning_curves/runs_symb_hpobench/parsimony{parsimony}/surr/{run_name}"

            df_error_metrics = pd.read_csv(f"{symb_dir}/error_metrics.csv")
            df_error_metrics["rmse_test"] = np.sqrt(df_error_metrics["mse_test"])
            df_error_metrics["rmse_train"] = np.sqrt(df_error_metrics["mse_train"])
            df_error_metrics = df_error_metrics[df_error_metrics["n_samples"] == n_samples]

            logger.info(f"Number of SR evaluations found: {len(df_error_metrics)}")

            df_complexity = pd.read_csv(f"{symb_dir}/complexity.csv")
            df_complexity = df_complexity[df_complexity["n_samples"] == n_samples]
            logger.info(f"Number of times complexity == -1: {len(df_complexity[df_complexity['program_operations'] == -1])}")
            # if len(df_complexity[df_complexity['program_operations'] == -1]) > 2:
            #     logger.info(f"Leave out {parsimony}, as complexity for more than 2 seeds could not be evaluated.")
            #     continue
            df_complexity = df_complexity[df_complexity["program_operations"] != -1]

            df_joined = pd.DataFrame({
                "rmse_test": [df_error_metrics["rmse_test"].mean(axis=0)],
                "complexity": [df_complexity["program_operations"].mean(axis=0)]
            })
            df_joined.insert(0, "Parsimony", parsimony)
            df_joined_all = pd.concat((df_joined_all, df_joined))

        df_joined_all['Parsimony'] = df_joined_all['Parsimony'].astype(str)
        df_joined_all.to_csv(f"{plot_dir}/df_joined_all_{run_name}")

        if i == 1:
            ind = 4
        elif i == 2:
            ind = 2
        elif i == 3:
            ind = 5
        elif i == 4:
            ind = 3
        else:
            ind = i + 1
        ax = plt.subplot(2, 3, ind)

        if ind == 2:
            plt.suptitle(f"Datset: {data_set}", fontsize=titlesize)

        if ind == 5:
            plt.suptitle(f"Datset: {data_set}")

        g = sns.scatterplot(data=df_joined_all, x="complexity", y="rmse_test", hue="Parsimony",
                            linestyles="", ax=ax, cmap="summer")
        sns.move_legend(g, "upper right", title="Parsimony", frameon=False, fontsize=labelsize)
        plt.title(f"{model_name} ({', '.join(optimized_parameters)})", fontsize=titlesize)
        plt.xlabel("Operation Count")
        plt.ylabel("RMSE $(c, s)$")
        plt.xlim(-0.5, 20.5)
        plt.xticks(np.arange(0, 22, 2.0), fontsize=labelsize)
        handles, labels = ax.get_legend_handles_labels()
        if i == 0:
            fig.legend(handles, labels, loc='center right', title="Parsimony")
        plt.legend([])

    plt.tight_layout(rect=(0, 0, 0.9, 1))
    plt.savefig(f"{plot_dir}/pointplot.png", dpi=400)
    plt.close()

