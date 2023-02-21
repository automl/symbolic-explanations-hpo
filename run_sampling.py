import logging
import os
import numpy as np
import pandas as pd
import time
import argparse
import dill as pickle
from smac import BlackBoxFacade, Callback
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
    n_samples = 200
    #model = "DT"
    model = functions[int(args.job_id)]
    data_sets = ["digits", "iris"]
    use_random_samples = True
    symb_reg = True

    if use_random_samples:
        run_type = "rand"
    else:
        run_type = "smac"

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

    if isinstance(model, NamedFunction):
        data_set_prefix = ""
        classifier = model
    else:
        hp_comb = combinations(hyperparams, 2)
        run_configs = []
        for hp_conf in hp_comb:
            for ds in data_sets:
                run_configs.append({hp_conf[0]: True, hp_conf[1]: True, "data_set_name": ds})
        run_conf = run_configs[int(job_id)]
        data_set_prefix = f"_{run_conf['data_set_name']}"
        if model == "MLP":
            classifier = MLP(**run_conf)
        elif model == "SVM":  # set lower tolerance, iris (stopping_criteria=0.00001)
            classifier = SVM(**run_conf)
        elif model == "BDT":
            classifier = BDT(**run_conf)
        elif model == "DT":
            classifier = DT(**run_conf)
        else:
            print(f"Unknown model: {model}")
            classifier = None

    function_name = classifier.name if isinstance(classifier, NamedFunction) else model

    # setup logging
    logger = logging.getLogger(__name__)

    optimized_parameters = classifier.configspace.get_hyperparameters()
    parameter_names = [param.name for param in optimized_parameters]

    sampling_run_name = f"{run_type}_{function_name.replace(' ', '_')}_{'_'.join(parameter_names)}" \
                        f"{data_set_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Start sampling for {sampling_run_name}.")

    if not os.path.exists("learning_curves/runs"):
        os.makedirs("learning_curves/runs")
    run_dir = f"learning_curves/runs/{sampling_run_name}"
    sampling_dir = f"{run_dir}/sampling"
    if not os.path.exists(sampling_dir):
        os.makedirs(sampling_dir)
    if not use_random_samples and not os.path.exists(f"{sampling_dir}/surrogates"):
        os.makedirs(f"{sampling_dir}/surrogates")

    with open(f"{sampling_dir}/classifier.pkl", "wb") as classifier_file:
        pickle.dump(classifier, classifier_file)

    df_samples = pd.DataFrame()

    for i in range(n_seeds):
        seed = i * 3

        np.random.seed(seed)

        if not isinstance(classifier, NamedFunction):
            classifier.set_seed(seed)

        logger.info(f"Sample configs and train {function_name} with seed {seed}.")

        if use_random_samples:
            configurations = classifier.configspace.sample_configuration(size=n_samples)
            performances = np.array(
                [classifier.train(config=x, seed=seed) for x in configurations]
            )
            configurations = np.array(
                [list(i.get_dictionary().values()) for i in configurations]
            )
        else:
            configurations, performances, _ = run_smac_optimization(
                configspace=classifier.configspace,
                facade=BlackBoxFacade,  # HyperparameterOptimizationFacade,
                target_function=classifier.train,
                function_name=function_name,
                n_eval=n_samples,
                run_dir=run_dir,
                seed=seed,
                callback=SurrogateModelCallback()
            )

        df = pd.DataFrame(
            data=np.concatenate((configurations,
                                 performances.reshape(-1, 1)), axis=1),
            columns=parameter_names + ["cost"])
        df.insert(0, "seed", seed)
        df_samples = pd.concat((df_samples, df))

    df_samples.to_csv(f"{sampling_dir}/samples.csv", index=False)

