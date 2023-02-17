import pandas as pd
import os
import sys
import logging
import dill as pickle
import matplotlib.pyplot as plt

from utils.functions import NamedFunction
from utils import functions


sys.modules['functions'] = functions


if __name__ == "__main__":
    sampling_run_name = "smac_Polynom_function_2D_X0_X1_20230216_200840"

    # setup logging
    logger = logging.getLogger(__name__)

    plot_dir = f"learning_curves/plots/{sampling_run_name}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    run_dir = f"learning_curves/runs/{sampling_run_name}"

    df_error_metrics = pd.read_csv(f"{run_dir}/error_metrics.csv")
    df_samples = pd.read_csv(f"{run_dir}/sampling/samples.csv")
    with open(f"{run_dir}/sampling/classifier.pkl", "rb") as classifier_file:
        classifier = pickle.load(classifier_file)
    if isinstance(classifier, NamedFunction):
        model_name = classifier.expression
    else:
        model_name = classifier.name

    avg_cost = df_samples.mean()["cost"]
    std_cost = df_samples.std()["cost"]

    # for each number of samples, average over seeds
    # df_avg_error_metrics = df_error_metrics.groupby("n_samples").mean().reset_index().add_prefix("mean_")
    # df_std_error_metrics = df_error_metrics.groupby("n_samples").std().reset_index().add_prefix("std_")
    # df_avg_std_error_metrics = pd.concat([df_avg_error_metrics, df_std_error_metrics], axis=1)
    # 
    # df_avg_std_error_metrics.plot(x="mean_n_samples", y="mean_mse_test_smac", yerr="std_mse_test_smac", linestyle="", marker="o")
    # plt.savefig(f"{run_dir}/test.png")

    df_error_metrics.boxplot("mse_test_smac", by="n_samples")
    plt.suptitle(model_name)
    plt.title(f"Function Value Avg: {avg_cost:.2f} / Std: {std_cost:.2f}", fontsize=10)
    plt.ylabel("Test MSE")
    plt.savefig(f"{plot_dir}/boxplot.png")
