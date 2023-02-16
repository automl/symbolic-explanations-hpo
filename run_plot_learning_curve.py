import pandas as pd
import logging
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sampling_run_name = "smac_Quadratic function A_x_20230216_092250"

    # setup logging
    logger = logging.getLogger(__name__)

    run_dir = f"learning_curves/runs/{sampling_run_name}"
    sampling_dir = f"{run_dir}/sampling"
    model = sampling_run_name.split("_")[0]

    df_error_metrics = pd.read_csv(f"{run_dir}/error_metrics.csv")

    # for each number of samples, average over seeds
    df_avg_error_metrics = df_error_metrics.groupby("n_samples").mean().reset_index().add_prefix("mean_")
    df_std_error_metrics = df_error_metrics.groupby("n_samples").std().reset_index().add_prefix("std_")
    df_avg_std_error_metrics = pd.concat([df_avg_error_metrics, df_std_error_metrics], axis=1)

    df_avg_std_error_metrics.plot(x="mean_n_samples", y="mean_mse_test_smac", yerr="std_mse_test_smac", linestyle="", marker="o")
    plt.savefig(f"{run_dir}/test.png")
