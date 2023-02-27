import pandas as pd
import os
import sys
import numpy as np
import logging
import dill as pickle

from utils.utils import get_hpo_test_data, plot_symb2d
from utils.functions import NamedFunction
from utils import functions


sys.modules['functions'] = functions


if __name__ == "__main__":
    model_name = "symb_best"
    n_samples = 100
    symb_seeds = [0, 3, 6]
    run_names = [
        "rand_DT_max_depth_min_samples_leaf_digits_20230221_114653",
        "smac_DT_max_depth_min_samples_leaf_digits_20230224_090309",
    ]

    # set up directories
    plot_dir = f"learning_curves/plots"
    viz_plot_dir = f"{plot_dir}/visualization"
    if not os.path.exists(viz_plot_dir):
        os.makedirs(viz_plot_dir)

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


        if "iris" in sampling_run_name:
            data_set = "Iris"
        elif "digits" in sampling_run_name:
            data_set = "Digits"
        else:
            data_set = None

        smac_run_name = [filename for filename in
                         os.listdir(f"learning_curves/runs/") if
                         filename.startswith(f"smac_{sampling_run_name}")][0]
        rand_run_name = [filename for filename in
                         os.listdir(f"learning_curves/runs/") if
                         filename.startswith(f"rand_{sampling_run_name}")][0]
        classifier_dir = f"learning_curves/runs/{smac_run_name}"
        symb_dir_smac = f"learning_curves/runs/{smac_run_name}/{model_name}/symb_models"
        symb_dir_rand = f"learning_curves/runs/{rand_run_name}/{model_name}/symb_models"

        with open(f"{classifier_dir}/sampling/classifier.pkl", "rb") as classifier_file:
            classifier = pickle.load(classifier_file)
        if isinstance(classifier, NamedFunction):
            classifier_name = classifier.name
        else:
            classifier_name = classifier.name
        optimized_parameters = classifier.configspace.get_hyperparameters()
        parameter_names = [param.name for param in optimized_parameters]

        df_samples_smac = pd.read_csv(f"{classifier_dir}/sampling/samples.csv")
        df_samples_rand = pd.read_csv(f"learning_curves/runs/{rand_run_name}/sampling/samples.csv")

        logger.info(f"Get test data for {classifier_name}.")
        X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, 100)

        symbolic_models = {}

        for sampling_seed in df_samples_smac.seed.unique():
            logger.info(f"Considering sampling seed {sampling_seed}.")
            df_sampling_seed_smac = df_samples_smac.copy()[df_samples_smac["seed"] == sampling_seed]
            df_sampling_seed_rand = df_samples_rand.copy()[df_samples_rand["seed"] == sampling_seed]

            X_train_smac = np.array(df_sampling_seed_smac[[parameter_names[0], parameter_names[1]]])[:n_samples]
            X_train_rand = np.array(df_sampling_seed_rand[[parameter_names[0], parameter_names[1]]])[:n_samples]

            for symb_seed in symb_seeds:
                logger.info(f"Considering symb seed {symb_seed}.")

                with open(f"{symb_dir_rand}/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl", "rb") as symb_model_file_rand:
                    symb_rand = pickle.load(symb_model_file_rand)
                with open(f"{symb_dir_smac}/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl", "rb") as symb_model_file_smac:
                    symb_smac = pickle.load(symb_model_file_smac)

                symbolic_models["Symb-smac"] = symb_smac
                symbolic_models["Symb-rand"] = symb_rand

                plot = plot_symb2d(
                                X_train_smac=X_train_smac.T,
                                X_train_compare=X_train_rand.T,
                                X_test=X_test,
                                y_test=y_test,
                                function_name=classifier.name,
                                metric_name="Test Error Rate",
                                symbolic_models=symbolic_models,
                                parameters=optimized_parameters,
                                plot_dir=viz_plot_dir,
                                filename=f"{model_name}_{'_'.join(parameter_names)}_{data_set}_n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}"
                            )
