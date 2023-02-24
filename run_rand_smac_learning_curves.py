import pandas as pd
import os
import sys
import logging
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.utils import get_hpo_test_data
from utils.functions import NamedFunction
from utils import functions


sys.modules['functions'] = functions


if __name__ == "__main__":
    model_name = "symb_best"
    run_names = [
        "rand_BDT_learning_rate_n_estimators_digits_20230221_114624",
        "rand_BDT_learning_rate_n_estimators_iris_20230221_114624",
        "rand_Branin_2D_X0_X1_20230221_120527",
        "rand_Camelback_2D_X0_X1_20230221_120527",
        "rand_DT_max_depth_min_samples_leaf_digits_20230221_114653",
        "rand_DT_max_depth_min_samples_leaf_iris_20230221_114651",
        "rand_Exponential_function_2D_X0_X1_20230221_120528",
        "rand_Linear_2D_X0_X1_20230221_120527",
        "rand_MLP_learning_rate_init_max_iter_digits_20230221_114330",
        "rand_MLP_learning_rate_init_max_iter_iris_20230221_114330",
        "rand_MLP_learning_rate_init_n_layer_digits_20230221_114329",
        "rand_MLP_learning_rate_init_n_layer_iris_20230221_114329",
        "rand_MLP_learning_rate_init_n_neurons_digits_20230221_114330",
        "rand_MLP_learning_rate_init_n_neurons_iris_20230221_114330",
        "rand_MLP_max_iter_n_layer_digits_20230221_114332",
        "rand_MLP_max_iter_n_layer_iris_20230221_114330",
        "rand_MLP_max_iter_n_neurons_digits_20230221_114332",
        "rand_MLP_max_iter_n_neurons_iris_20230221_114330",
        "rand_MLP_n_layer_n_neurons_digits_20230221_114330",
        "rand_MLP_n_layer_n_neurons_iris_20230221_114330",
        "rand_Polynom_function_2D_X0_X1_20230221_120528",
        "rand_Rosenbrock_2D_X0_X1_20230221_120527",
        "rand_SVM_C_coef0_digits_20230221_114754",
        "rand_SVM_C_coef0_iris_20230221_114755",
        "rand_SVM_C_degree_digits_20230221_114756",
        "rand_SVM_C_degree_iris_20230221_114756",
        "rand_SVM_C_gamma_digits_20230221_114754",
        "rand_SVM_C_gamma_iris_20230221_114755",
        "rand_SVM_coef0_degree_digits_20230221_114754",
        "rand_SVM_coef0_degree_iris_20230221_114756",
        "rand_SVM_coef0_gamma_digits_20230221_114755",
        "rand_SVM_coef0_gamma_iris_20230221_114755",
        "rand_SVM_degree_gamma_digits_20230221_114754",
        "rand_SVM_degree_gamma_iris_20230221_114754",
        "smac_BDT_learning_rate_n_estimators_digits_20230223_162320",
        "smac_BDT_learning_rate_n_estimators_iris_20230223_162320",
        "smac_Branin_2D_X0_X1_20230223_162155",
        "smac_Camelback_2D_X0_X1_20230223_162155",
        "smac_DT_max_depth_min_samples_leaf_digits_20230224_090309",
        "smac_DT_max_depth_min_samples_leaf_iris_20230224_090310",
        "smac_Exponential_function_2D_X0_X1_20230223_162156",
        "smac_Linear_2D_X0_X1_20230223_162155",
        "smac_MLP_learning_rate_init_max_iter_digits_20230223_162437",
        "smac_MLP_learning_rate_init_max_iter_iris_20230223_162436",
        "smac_MLP_learning_rate_init_n_layer_digits_20230223_162436",
        "smac_MLP_learning_rate_init_n_layer_iris_20230223_162436",
        "smac_MLP_learning_rate_init_n_neurons_digits_20230223_162436",
        "smac_MLP_learning_rate_init_n_neurons_iris_20230223_162436",
        "smac_MLP_max_iter_n_layer_digits_20230223_162436",
        "smac_MLP_max_iter_n_layer_iris_20230223_162436",
        "smac_MLP_max_iter_n_neurons_digits_20230223_162436",
        "smac_MLP_max_iter_n_neurons_iris_20230223_162436",
        "smac_MLP_n_layer_n_neurons_digits_20230223_162436",
        "smac_MLP_n_layer_n_neurons_iris_20230223_162437",
        "smac_Polynom_function_2D_X0_X1_20230223_162156",
        "smac_Rosenbrock_2D_X0_X1_20230223_162155",
        "smac_SVM_C_coef0_digits_20230223_164415",
        "smac_SVM_C_coef0_iris_20230223_162859",
        "smac_SVM_C_degree_digits_20230223_162900",
        "smac_SVM_C_degree_iris_20230223_164415",
        "smac_SVM_C_gamma_digits_20230223_162900",
        "smac_SVM_C_gamma_iris_20230223_162859",
        "smac_SVM_coef0_degree_digits_20230223_162859",
        "smac_SVM_coef0_degree_iris_20230223_162859",
        "smac_SVM_coef0_gamma_digits_20230223_162859",
        "smac_SVM_coef0_gamma_iris_20230223_162859",
        "smac_SVM_degree_gamma_digits_20230223_162900",
        "smac_SVM_degree_gamma_iris_20230223_162859"
    ]

    # set up directories
    plot_dir = f"learning_curves/plots/combined_plots_smac_rand_best"
    complexity_plot_dir = f"{plot_dir}/complexity"
    mse_plot_dir = f"{plot_dir}/mse"
    rmse_plot_dir = f"{plot_dir}/rmse"
    if not os.path.exists(complexity_plot_dir):
        os.makedirs(complexity_plot_dir)
    if not os.path.exists(mse_plot_dir):
        os.makedirs(mse_plot_dir)
    if not os.path.exists(rmse_plot_dir):
        os.makedirs(rmse_plot_dir)

    # setup logging
    logger = logging.getLogger(__name__)
    handler2 = logging.StreamHandler()
    handler2.setLevel("INFO")
    handler2.setFormatter(
        logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    handler2.setStream(sys.stdout)
    logger.root.addHandler(handler2)
    logger.root.setLevel("INFO")

    logger.info(f"Save plots to {plot_dir}.")

    run_names_cut = ["_".join(run.split("_")[1:-2]) for run in run_names]
    run_names_cut = set(run_names_cut)

    for sampling_run_name in run_names_cut:

        logger.info(f"Create plot for {sampling_run_name}.")

        smac_run_name = [filename for filename in
                         os.listdir(f"learning_curves/runs/") if
                         filename.startswith(f"smac_{sampling_run_name}")][0]
        rand_run_name = [filename for filename in
                         os.listdir(f"learning_curves/runs/") if
                         filename.startswith(f"rand_{sampling_run_name}")][0]
        classifier_dir = f"learning_curves/runs/{smac_run_name}"

        with open(f"{classifier_dir}/sampling/classifier.pkl", "rb") as classifier_file:
            classifier = pickle.load(classifier_file)
        if isinstance(classifier, NamedFunction):
            classifier_name = classifier.name
        else:
            classifier_name = classifier.name
        optimized_parameters = classifier.configspace.get_hyperparameters()
        parameter_names = [param.name for param in optimized_parameters]

        logger.info(f"Get test data for {classifier_name}.")
        X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, 100)

        avg_cost = y_test.mean()
        std_cost = y_test.std()

        df_error_metrics_all = pd.DataFrame()
        df_complexity_all = pd.DataFrame()

        for sampling_type in ["Symbolic Regression (Random)", "Symbolic Regression (SMAC)", "Gaussian Process (SMAC)"]:

            if sampling_type == "Symbolic Regression (SMAC)" or sampling_type == "Gaussian Process (SMAC)":
                run_dir = f"learning_curves/runs/{smac_run_name}"
            else:
                run_dir = f"learning_curves/runs/{rand_run_name}"

            model_dir = f"{run_dir}/{model_name}"

            if sampling_type == "Gaussian Process (SMAC)":
                df_error_metrics = pd.read_csv(f"{run_dir}/surrogate_error_metrics.csv")
            else:
                df_error_metrics = pd.read_csv(f"{model_dir}/error_metrics.csv")
                df_complexity = pd.read_csv(f"{model_dir}/complexity.csv")
                df_complexity.insert(0, "Experiment", f"{sampling_type}")
                df_complexity_all = pd.concat((df_complexity_all, df_complexity))

            df_error_metrics["rmse_test_smac"] = np.sqrt(df_error_metrics["mse_test_smac"])
            df_error_metrics["rmse_train_smac"] = np.sqrt(df_error_metrics["mse_train_smac"])
            df_error_metrics.insert(0, "Experiment", f"{sampling_type}")
            df_error_metrics_all = pd.concat((df_error_metrics_all, df_error_metrics))

        logger.info(f"Create plots.")

        # Plot RMSE
        plt.figure()
        _, ax = plt.subplots(figsize=(8, 5))
        sns.pointplot(data=df_error_metrics_all, x="n_samples", y="rmse_test_smac", hue="Experiment")#, showfliers=False)
        plt.title(f"{classifier_name}: {', '.join(parameter_names)}")
        #plt.title(f"Test Mean: {avg_cost:.3f}, Test Std.: {std_cost:.3f}", fontsize=10),
        plt.ylabel("Test RMSE")
        plt.xlabel("Number of Samples")
        plt.axhline(y=std_cost, color='darkred', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=(0, 0.05, 1, 1))
        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.45, -0.23),
            ncol=3,
            title=None, frameon=False,
        )
        plt.savefig(f"{rmse_plot_dir}/{sampling_run_name}_pointplot.png", dpi=200)

        # Plot Complexity
        plt.figure()
        _, ax = plt.subplots(figsize=(8, 5))
        sns.pointplot(data=df_complexity_all, x="n_samples", y="complexity", hue="Experiment")#, showfliers=False)
        plt.suptitle(f"{classifier_name}: {', '.join(parameter_names)}")
       # plt.title("Symbolic Regression Program Length")
        plt.ylabel("Program Length")
        plt.tight_layout(rect=(0, 0.05, 1, 1))
        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.45, -0.23),
            ncol=2,
            title=None, frameon=False,
        )
        plt.tight_layout()
        plt.savefig(f"{complexity_plot_dir}/{sampling_run_name}_complexity_pointplot.png", dpi=200)
