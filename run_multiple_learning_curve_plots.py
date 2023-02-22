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
    rand = True
    model_names = ["symb"]#, "surrogate"]
    symb_dir_postfixes = ["_defaults", "_fixed_const_range"]
    run_names = [
        "rand_BDT_learning_rate_n_estimators_digits_20230221_114624",
        "rand_MLP_max_iter_n_neurons_iris_20230221_114330",
        "rand_BDT_learning_rate_n_estimators_iris_20230221_114624",
        "rand_MLP_n_layer_n_neurons_digits_20230221_114330",
        "rand_Branin_2D_X0_X1_20230221_120527",
        "rand_MLP_n_layer_n_neurons_iris_20230221_114330",
        "rand_Camelback_2D_X0_X1_20230221_120527",
        "rand_Polynom_function_2D_X0_X1_20230221_120528",
        "rand_DT_max_depth_min_samples_leaf_digits_20230221_114653",
        "rand_Rosenbrock_2D_X0_X1_20230221_120527",
        "rand_DT_max_depth_min_samples_leaf_iris_20230221_114651",
        # "rand_SVM_C_coef0_digits_20230221_114754",
        "rand_Exponential_function_2D_X0_X1_20230221_120528",
        "rand_SVM_C_coef0_iris_20230221_114755",
        "rand_Linear_2D_X0_X1_20230221_120527",
        # "rand_SVM_C_degree_digits_20230221_114756",
        "rand_MLP_learning_rate_init_max_iter_digits_20230221_114330",
        "rand_SVM_C_degree_iris_20230221_114756",
        "rand_MLP_learning_rate_init_max_iter_iris_20230221_114330",
        "rand_SVM_C_gamma_digits_20230221_114754",
        "rand_MLP_learning_rate_init_n_layer_digits_20230221_114329",
        "rand_SVM_C_gamma_iris_20230221_114755",
        "rand_MLP_learning_rate_init_n_layer_iris_20230221_114329",
        # "rand_SVM_coef0_degree_digits_20230221_114754",
        "rand_MLP_learning_rate_init_n_neurons_digits_20230221_114330",
        # "rand_SVM_coef0_degree_iris_20230221_114756",
        "rand_MLP_learning_rate_init_n_neurons_iris_20230221_114330",
        "rand_SVM_coef0_gamma_digits_20230221_114755",
        "rand_MLP_max_iter_n_layer_digits_20230221_114332",
        "rand_SVM_coef0_gamma_iris_20230221_114755",
        "rand_MLP_max_iter_n_layer_iris_20230221_114330",
        "rand_SVM_degree_gamma_digits_20230221_114754",
        "rand_MLP_max_iter_n_neurons_digits_20230221_114332",
        "rand_SVM_degree_gamma_iris_20230221_114754",
        # "smac_Linear_2D_X0_X1_20230216_200839",
        # "smac_Branin_2D_X0_X1_20230216_202959",
        # "smac_MLP_learning_rate_init_max_iter_iris_20230218_134148",
        # "smac_MLP_n_layer_n_neurons_digits_20230218_140256",
        # "smac_SVM_C_coef0_digits_20230218_124032",
        # "smac_SVM_coef0_degree_digits_20230218_124029",
        # "smac_SVM_coef0_degree_iris_20230218_124031",
        # "smac_SVM_coef0_gamma_digits_20230218_124031",
        # "smac_BDT_learning_rate_n_estimators_digits_20230218_141037",
        # "smac_BDT_learning_rate_n_estimators_iris_20230218_123429",
        # "smac_Camelback_2D_X0_X1_20230216_202959",
        # "smac_DT_max_depth_min_samples_leaf_digits_20230218_103755",
        # "smac_DT_max_depth_min_samples_leaf_iris_20230218_103751",
        # "smac_Exponential_function_2D_X0_X1_20230216_202958",
        # "smac_MLP_learning_rate_init_n_layer_digits_20230218_145001",
        # "smac_MLP_learning_rate_init_n_layer_iris_20230218_134149",
        # "smac_MLP_learning_rate_init_n_neurons_digits_20230218_145001",
        # "smac_MLP_learning_rate_init_n_neurons_iris_20230218_134149",
        # "smac_MLP_max_iter_n_layer_digits_20230218_134149",
        # "smac_MLP_max_iter_n_layer_iris_20230218_134146",
        # "smac_Polynom_function_2D_X0_X1_20230216_200840",
        # "smac_Rosenbrock_2D_X0_X1_20230216_202959",
        # "smac_SVM_C_gamma_digits_20230218_124031",
    ]

    # setup logging
    logger = logging.getLogger(__name__)

    # set up directories
    plot_dir = f"learning_curves/plots/combined_plots"
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

    for sampling_run_name in run_names:
        run_dir = f"learning_curves/runs/{sampling_run_name}"
        logger.info(f"Create plot for {sampling_run_name}.")

        with open(f"{run_dir}/sampling/classifier.pkl", "rb") as classifier_file:
            classifier = pickle.load(classifier_file)
        if isinstance(classifier, NamedFunction):
            classifier_name = classifier.name
        else:
            classifier_name = classifier.name
        optimized_parameters = classifier.configspace.get_hyperparameters()
        parameter_names = [param.name for param in optimized_parameters]

        X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, 100)

        avg_cost = y_test.mean()
        std_cost = y_test.std()

        df_error_metrics_all = pd.DataFrame()
        for symb_dir_postfix in symb_dir_postfixes:
            for model_name in model_names:

                if model_name == "surrogate" and symb_dir_postfix != "":
                    continue

                logger.info(f"Get error metrics for {model_name}{symb_dir_postfix}.")

                model_dir = f"{run_dir}/{model_name}{symb_dir_postfix}"

                if model_name == "surrogate":
                    df_error_metrics = pd.read_csv(f"{model_dir}/surrogate_error_metrics.csv")
                else:
                    df_error_metrics = pd.read_csv(f"{model_dir}/error_metrics.csv")

                df_error_metrics["rmse_test_smac"] = np.sqrt(df_error_metrics["mse_test_smac"])
                df_error_metrics["rmse_train_smac"] = np.sqrt(df_error_metrics["mse_train_smac"])
                df_error_metrics.insert(0, "exp_name", f"{model_name}{symb_dir_postfix}")
                df_error_metrics_all = pd.concat((df_error_metrics_all, df_error_metrics))

        logger.info(f"Create boxplot.")

        # Plot RMSE (Boxplot)
        sns.boxplot(data=df_error_metrics_all, x="n_samples", y="rmse_test_smac", hue="exp_name")
        #df_error_metrics.boxplot("rmse_test_smac", by="n_samples")
        plt.suptitle(f"{classifier_name}: {', '.join(parameter_names)}")
        plt.title(f"Function Value Avg: {avg_cost:.2f} / Std: {std_cost:.2f}", fontsize=10),
        plt.ylabel("Test RMSE")
        #plt.gca().set_ylim(top=2*std_cost)
        plt.tight_layout()

        plt.savefig(f"{plot_dir}/{sampling_run_name}_boxplot.png", dpi=200)