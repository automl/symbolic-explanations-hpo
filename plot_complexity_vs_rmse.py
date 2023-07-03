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

    labelsize = 16
    titlesize=18

    perform_elbow = True
    elbow_type = "increase_limit"  # "increase_limit", "parallel_lines", "parallel_lines_curve"

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

        if i == 2:
            plt.figtext(0.5, 0.98, f"Dataset: {data_set}", ha="center", va="top", fontsize=titlesize)
        if i == 3:
            plt.figtext(0.5, 0.50, f"Dataset: {data_set}", ha="center", va="top", fontsize=titlesize)

        g = sns.scatterplot(data=df_joined_all, x="complexity", y="rmse_test", hue="Parsimony",
                            linestyles="", s=80, ax=ax, palette="cividis")
        if model_name == "LR":
            classifier_title = "Logistic Regression"
        else:
            classifier_title = model_name
        plt.title(f"{classifier_title} ({', '.join(optimized_parameters)})", fontsize=titlesize)
        if ind > 3:
            plt.xlabel("Operation Count", fontsize=labelsize, labelpad=10)
        else:
            plt.xlabel("")
        if ind == 1 or ind == 4:
            plt.ylabel("RMSE $(c, s)$", fontsize=labelsize, labelpad=14)
        else:
            plt.ylabel("")
        plt.xlim(-0.5, 20.5)
        plt.yticks(fontsize=labelsize-2)
        plt.xticks(np.arange(0, 24, 4.0), fontsize=labelsize-2)
        plt.legend([], [], frameon=False)

        if perform_elbow:
            df_sorted = df_joined_all.sort_values('complexity')
            df_sorted = df_sorted.sort_values('rmse_test')
            x_lower_right = df_sorted.iloc[0]["complexity"]
            x_upper_left = df_sorted.iloc[-1]["complexity"]
            selected_point = None

            if elbow_type == "increase_limit":
                for index in reversed(range(len(df_sorted) - 1)):
                    if df_sorted.iloc[index - 1]["rmse_test"] > 0.9 * df_sorted.iloc[index]["rmse_test"] and \
                            df_sorted.iloc[index - 1]["complexity"] - df_sorted.iloc[index]["complexity"] > 2:
                        selected_point = df_sorted.iloc[index]
                        break
                if selected_point is not None:
                    plt.plot(selected_point["complexity"], selected_point["rmse_test"], 'P', color='red')

            elif elbow_type == "parallel_lines":
                y_lower_right = df_sorted.iloc[0]["rmse_test"]
                y_upper_left = df_sorted.iloc[-1]["rmse_test"]
                plt.plot([x_upper_left, x_lower_right], [y_upper_left, y_lower_right], color='red')

            elif elbow_type == "parallel_lines_curve":
                from scipy.optimize import curve_fit
                from sklearn.metrics import r2_score

                def objective(x, p_a, p_b):
                    return p_a * (x+0.1)**(-p_b)

                popt, _ = curve_fit(f=objective, xdata=df_joined_all["complexity"].values,
                                    ydata=df_joined_all["rmse_test"].values)
                a, b = popt

                print(data_set)
                print(classifier_title)
                print(r2_score(df_joined_all["rmse_test"].values, objective(df_joined_all["complexity"].values, a, b)))

                y_lower_right = objective(df_sorted.iloc[0]["complexity"], a, b)
                y_upper_left = objective(df_sorted.iloc[-1]["complexity"], a, b)

                complexity_ls = np.linspace(x_upper_left, x_lower_right, 100)
                plt.plot(complexity_ls, objective(complexity_ls, a, b), color="black")
                plt.plot([x_upper_left, x_lower_right], [y_upper_left, y_lower_right], color='orange')
            else:
                    raise ValueError(f"Unknown elbow type: {elbow_type}")
            if elbow_type in ("parallel_lines", "parallel_lines_curve"):
                slope = (y_lower_right - y_upper_left) / (x_lower_right - x_upper_left)
                selected_point = None

                for index1, row1 in df_joined_all.iterrows():
                    points_below = False
                    for index2, row2 in df_joined_all.iterrows():
                        line_y = slope * (row2["complexity"] - row1["complexity"]) + row1["rmse_test"]
                        if row2['rmse_test'] < line_y:
                            points_below = True
                            break
                    if not points_below:
                        selected_point = row1
                        break

                if selected_point is not None:
                    line_start_y = slope * (x_upper_left - selected_point["complexity"]) + selected_point["rmse_test"]
                    line_end_x = (y_lower_right - selected_point["rmse_test"]) / slope + selected_point["complexity"]
                    plt.plot([x_upper_left, line_end_x], [line_start_y, y_lower_right], color='red')

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='center right', title="Parsimony", frameon=False, fontsize=titlesize)
    legend.get_title().set_fontsize(titlesize)
    for handle in legend.legendHandles:
        handle.set_sizes([60])
    plt.tight_layout(rect=(0, 0, 0.82, 0.95), h_pad=5)
    plt.savefig(f"{plot_dir}/pointplot.png", dpi=400)
    plt.close()

