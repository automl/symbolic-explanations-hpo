import logging
import os
import numpy as np
import pandas as pd
import time
import argparse
import dill as pickle
from smac import BlackBoxFacade, Callback
from sklearn import datasets
from itertools import combinations

from utils.smac_utils import run_smac_optimization
from utils.model_wrapper import SVM, MLP, BDT, DT
from utils.functions import get_functions2d, NamedFunction


class SurrogateModelCallback(Callback):
    def on_next_configurations_end(self, config, config_selector):
        if config._acquisition_function._eta:
            surrogate_model = config._model
            processed_configs = len(config._processed_configs)
            with open(f"{sampling_dir}/surrogates/seed{seed}_samples{processed_configs}.pkl", "wb") as surrogate_file:
                pickle.dump(surrogate_model, surrogate_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    args = parser.parse_args()
    job_id = args.job_id

    functions = get_functions2d()
    n_seeds = 5
    n_smac_samples = 200
    model = "BDT"
    #model = functions[int(args.job_id)]
    symb_reg = True

    if model == "MLP":
        hyperparams = [
            "optimize_n_neurons",
            "optimize_n_layer",
            "optimize_learning_rate_init",
            "optimize_max_iter"
        ]
    elif model == "SVM":
        hyperparams = [
            "optimize_C",
            "optimize_degree",
            "optimize_coef",
            "optimize_gamma"
        ]
    elif model == "BDT":
        hyperparams = [
            "optimize_learning_rate", "optimize_n_estimators"
        ]
    elif model == "DT":
        hyperparams = [
            "optimize_max_depth", "optimize_min_samples_leaf"
        ]
    else:
        hyperparams = None

    hp_comb = combinations(hyperparams, 2)
    config_list = []
    for hp_conf in hp_comb:
        for dataset in [datasets.load_digits(), datasets.load_iris()]:
            config_list.append({hp_conf[0]: True, hp_conf[1]: True, "data_set": dataset})
    hp_data_conf = config_list[int(job_id)]

    if model == "MLP":
        classifier = MLP(**hp_data_conf)
    elif model == "SVM":  # set lower tolerance, iris (stopping_criteria=0.00001)
        classifier = SVM(**hp_data_conf)
    elif model == "BDT":
        classifier = BDT(**hp_data_conf)
    elif model == "DT":
        classifier = DT(**hp_data_conf)
    elif isinstance(model, NamedFunction):
        classifier = model
    else:
        print(f"Unknown model: {model}")
        classifier = None

    function_name = classifier.name if isinstance(classifier, NamedFunction) else model

    # setup logging
    logger = logging.getLogger(__name__)

    optimized_parameters = classifier.configspace.get_hyperparameters()
    parameter_names = [param.name for param in optimized_parameters]

    if not os.path.exists("learning_curves/runs"):
        os.makedirs("learning_curves/runs")
    run_dir = f"learning_curves/runs/smac_{function_name.replace(' ', '_')}_{'_'.join(parameter_names)}_" \
              f"{time.strftime('%Y%m%d_%H%M%S')}"
    sampling_dir = f"{run_dir}/sampling"
    if not os.path.exists(sampling_dir):
        os.makedirs(sampling_dir)
    if not os.path.exists(f"{sampling_dir}/surrogates"):
        os.makedirs(f"{sampling_dir}/surrogates")

    with open(f"{sampling_dir}/classifier.pkl", "wb") as classifier_file:
        pickle.dump(classifier, classifier_file)

    df_smac_samples = pd.DataFrame()

    for i in range(n_seeds):
        seed = i * 3

        np.random.seed(seed)

        if not isinstance(classifier, NamedFunction):
            classifier.set_seed(seed)

        logger.info(f"Run SMAC to sample configs and train {function_name} with seed {seed}.")

        smac_configurations, smac_performances, smac_facade = run_smac_optimization(
            configspace=classifier.configspace,
            facade=BlackBoxFacade,  # HyperparameterOptimizationFacade,
            target_function=classifier.train,
            function_name=function_name,
            n_eval=n_smac_samples,
            run_dir=run_dir,
            seed=seed,
            callback=SurrogateModelCallback()
        )

        df = pd.DataFrame(
            data=np.concatenate((smac_configurations.T,
                                 smac_performances.reshape(n_smac_samples, 1)), axis=1),
            columns=parameter_names + ["cost"])
        df.insert(0, "seed", seed)

        df_smac_samples = pd.concat((df_smac_samples, df))

    df_smac_samples.to_csv(f"{sampling_dir}/samples.csv", index=False)

