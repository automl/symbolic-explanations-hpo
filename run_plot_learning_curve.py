import pandas as pd
import os
import sys
import logging
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_hpo_test_data
from utils.functions import NamedFunction
from utils import functions


sys.modules['functions'] = functions


if __name__ == "__main__":
    rand = False
    evaluate_surrogate = False
    symb_dir_postfix = ""
    run_names = [
        "smac_Linear_2D_X0_X1_20230216_200839",
        "smac_Branin_2D_X0_X1_20230216_202959",
        "smac_MLP_learning_rate_init_max_iter_iris_20230218_134148",
        "smac_MLP_n_layer_n_neurons_digits_20230218_140256",
        "smac_SVM_C_coef0_digits_20230218_124032",
        "smac_SVM_coef0_degree_digits_20230218_124029",
        "smac_SVM_coef0_degree_iris_20230218_124031",
        "smac_SVM_coef0_gamma_digits_20230218_124031",
        "smac_BDT_learning_rate_n_estimators_digits_20230218_141037",
        "smac_BDT_learning_rate_n_estimators_iris_20230218_123429",
        "smac_Camelback_2D_X0_X1_20230216_202959",
        "smac_DT_max_depth_min_samples_leaf_digits_20230218_103755",
        "smac_DT_max_depth_min_samples_leaf_iris_20230218_103751",
        "smac_Exponential_function_2D_X0_X1_20230216_202958",
        "smac_MLP_learning_rate_init_n_layer_digits_20230218_145001",
        "smac_MLP_learning_rate_init_n_layer_iris_20230218_134149",
        "smac_MLP_learning_rate_init_n_neurons_digits_20230218_145001",
        "smac_MLP_learning_rate_init_n_neurons_iris_20230218_134149",
        "smac_MLP_max_iter_n_layer_digits_20230218_134149",
        "smac_MLP_max_iter_n_layer_iris_20230218_134146",
        "smac_Polynom_function_2D_X0_X1_20230216_200840",
        "smac_Rosenbrock_2D_X0_X1_20230216_202959",
        "smac_SVM_C_gamma_digits_20230218_124031",
    ]

    # setup logging
    logger = logging.getLogger(__name__)

    for sampling_run_name in run_names:
        # setup logging
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(filename=f"learning_curves/runs/{sampling_run_name}/plot_log.log",
                                      encoding="utf8")
        handler.setLevel("INFO")
        handler.setFormatter(
            logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
        )
        logger.root.addHandler(handler)
        handler2 = logging.StreamHandler()
        handler2.setLevel("INFO")
        handler2.setFormatter(
            logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
        )
        handler2.setStream(sys.stdout)
        logger.root.addHandler(handler2)
        logger.root.setLevel("INFO")

        logger.info(f"Create plots for {sampling_run_name}.")

        run_dir = f"learning_curves/runs/{sampling_run_name}"
        symb_dir = f"{run_dir}/symb{symb_dir_postfix}"
        plot_dir = f"learning_curves/plots"
        complexity_plot_dir = f"learning_curves/plots/complexity{symb_dir_postfix}"
        if evaluate_surrogate:
            mse_plot_dir = f"learning_curves/plots/surrogate_mse{symb_dir_postfix}"
            rmse_plot_dir = f"learning_curves/plots/surrogate_rmse{symb_dir_postfix}"
        else:
            mse_plot_dir = f"learning_curves/plots/mse{symb_dir_postfix}"
            rmse_plot_dir = f"learning_curves/plots/rmse{symb_dir_postfix}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        if not os.path.exists(complexity_plot_dir):
            os.makedirs(complexity_plot_dir)
        if not os.path.exists(mse_plot_dir):
            os.makedirs(mse_plot_dir)
        if not os.path.exists(rmse_plot_dir):
            os.makedirs(rmse_plot_dir)

        if evaluate_surrogate:
            df_error_metrics = pd.read_csv(f"{symb_dir}/surrogate_error_metrics.csv")
        else:
            df_error_metrics = pd.read_csv(f"{symb_dir}/error_metrics.csv")
        with open(f"{run_dir}/sampling/classifier.pkl", "rb") as classifier_file:
            classifier = pickle.load(classifier_file)
        if isinstance(classifier, NamedFunction):
            model_name = classifier.name
        else:
            model_name = classifier.name
        optimized_parameters = classifier.configspace.get_hyperparameters()
        parameter_names = [param.name for param in optimized_parameters]

        X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, 100)

        avg_cost = y_test.mean()
        std_cost = y_test.std()

        df_all_complexity = pd.DataFrame()
        df_error_metrics["rmse_test_smac"] = np.sqrt(df_error_metrics["mse_test_smac"])
        df_error_metrics["rmse_train_smac"] = np.sqrt(df_error_metrics["mse_train_smac"])

        if not evaluate_surrogate:
            for n_samples in df_error_metrics.n_samples.unique():
                for sampling_seed in df_error_metrics.sampling_seed.unique():
                    for symb_seed in df_error_metrics.symb_seed.unique():
                        print(
                            f"Evaluate complexity for n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}")
                        try:
                            with open(
                                    f"{symb_dir}/symb_models/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl",
                                    "rb") as symb_model_file:
                                symb_model = pickle.load(symb_model_file)
                                complexity = symb_model._program.length_
                                df_complexity = pd.DataFrame({
                                    "complexity": [complexity],
                                    "n_samples": [n_samples],
                                    "sampling_seed": [sampling_seed],
                                    "symb_seed": [symb_seed]
                                })
                                df_all_complexity = pd.concat((df_all_complexity, df_complexity))
                        except:
                            continue

            # for each number of samples, average over seeds
            df_avg_error_complexity = df_all_complexity.groupby("n_samples").mean().reset_index()
            df_std_error_complexity = df_all_complexity.groupby("n_samples").std().reset_index()
            df_avg_std_complexity = pd.merge(left=df_avg_error_complexity, right=df_std_error_complexity, on="n_samples",
                                                suffixes=("_mean", "_std"))

        # for each number of samples, average over seeds
        df_avg_error_metrics = df_error_metrics.groupby("n_samples").mean().reset_index()
        df_std_error_metrics = df_error_metrics.groupby("n_samples").std().reset_index()
        df_avg_std_error_metrics = pd.merge(left=df_avg_error_metrics, right=df_std_error_metrics, on="n_samples",
                                            suffixes=("_mean", "_std"))

        if rand:
            df_avg_std_error_metrics = df_avg_std_error_metrics.rename(columns={
                "mse_test_smac_mean": "mse_test_mean",
                "mse_test_smac_std": "mse_test_std",
                "rmse_test_smac_mean": "rmse_test_mean",
                "rmse_test_smac_std": "rmse_test_std",
            })
            postfix = ""
        else:
            postfix = "_smac"

        logger.info(f"Save plots to {mse_plot_dir}.")

        # Plot MSE (Mean + Std)
        # df_avg_std_error_metrics.plot(x="n_samples", y=f"mse_test{postfix}_mean", yerr=f"mse_test{postfix}_std",
        #                               linestyle="", marker="o")
        # plt.suptitle(f"{model_name}: {', '.join(parameter_names)}")
        # plt.title(f"Function Value Avg: {avg_cost:.2f} / Std: {std_cost:.2f}", fontsize=10)
        # plt.ylabel("Test MSE")
        # plt.tight_layout()
        # plt.savefig(f"{mse_plot_dir}/{sampling_run_name}_mean_std_plot.png", dpi=200)

        # # mse_q3 = np.percentile(df_error_metrics.mse_test_smac, 75)
        # # mse_q1 = np.percentile(df_error_metrics.mse_test_smac, 25)
        # # iqr = mse_q3 - mse_q1
        # # cut_off = mse_q3 + 10*iqr
        # # df_outliers = df_error_metrics[df_error_metrics.mse_test_smac > cut_off]
        # # df_error_metrics = df_error_metrics.drop(df_outliers.index)

        # Plot MSE (Boxplot)
        df_error_metrics.boxplot("mse_test_smac", by="n_samples")
        plt.suptitle(f"{model_name}: {', '.join(parameter_names)}")
        plt.title(f"Function Value Avg: {avg_cost:.2f} / Std: {std_cost:.2f}", fontsize=10),# n_outliers={len(df_outliers.index)}", fontsize=10)
        plt.ylabel("Test MSE")
        plt.tight_layout()
        plt.savefig(f"{mse_plot_dir}/{sampling_run_name}_boxplot.png", dpi=200)

        logger.info(f"Save plots to {rmse_plot_dir}.")

        # Plot RMSE (Mean + Std)
        # df_avg_std_error_metrics.plot(x="n_samples", y=f"rmse_test{postfix}_mean", yerr=f"rmse_test{postfix}_std",
        #                               linestyle="", marker="o")
        # plt.suptitle(f"{model_name}: {', '.join(parameter_names)}")
        # plt.title(f"Function Value Avg: {avg_cost:.2f} / Std: {std_cost:.2f}", fontsize=10)
        # plt.ylabel("Test RMSE")
        # plt.tight_layout()
        # plt.savefig(f"{rmse_plot_dir}/{sampling_run_name}_mean_std_plot.png", dpi=200)

        # Plot RMSE (Boxplot)
        df_error_metrics.boxplot("rmse_test_smac", by="n_samples")
        plt.suptitle(f"{model_name}: {', '.join(parameter_names)}")
        plt.title(f"Function Value Avg: {avg_cost:.2f} / Std: {std_cost:.2f}", fontsize=10),# n_outliers={len(df_outliers.index)}", fontsize=10)
        plt.ylabel("Test RMSE")
        plt.tight_layout()
        plt.savefig(f"{rmse_plot_dir}/{sampling_run_name}_boxplot.png", dpi=200)

        if not evaluate_surrogate:
            logger.info(f"Save plots to {complexity_plot_dir}.")

            # Plot Complexity (Mean + Std)
            # df_avg_std_complexity.plot(x="n_samples", y="complexity_mean", yerr="complexity_std",
            #                               linestyle="", marker="o")
            # plt.suptitle(f"{model_name}: {', '.join(parameter_names)}")
            # plt.title("Complexity")
            # plt.ylabel("Program Length")
            # plt.tight_layout()
            # plt.savefig(f"{complexity_plot_dir}/{sampling_run_name}_complexity_mean_std_plot.png", dpi=200)

            # Plot Complexity (Boxplot)
            df_all_complexity.boxplot("complexity", by="n_samples")
            plt.suptitle(f"{model_name}: {', '.join(parameter_names)}")
            plt.title("Complexity")
            plt.ylabel("Program Length")
            plt.tight_layout()
            plt.savefig(f"{complexity_plot_dir}/{sampling_run_name}_complexity_boxplot.png", dpi=200)