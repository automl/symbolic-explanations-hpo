import pandas as pd
import sys
import numpy as np
import dill as pickle
from itertools import combinations

from utils.utils import get_hpo_test_data, get_scores, get_surrogate_predictions
from utils.functions_utils import get_functions2d, NamedFunction
from utils.model_utils import get_hyperparams, get_classifier_from_run_conf
from utils import functions_utils
from utils.logging_utils import get_logger


sys.modules['functions'] = functions_utils


N_SAMPLES_SPACING = np.linspace(20, 200, 10, dtype=int).tolist()


if __name__ == "__main__":
    n_test_samples = 100
    symb_seeds = [0, 3, 6]
    symb_dir_name = "mult_testeval_add_func"
    functions = get_functions2d()
    models = ["MLP", "SVM", "BDT", "DT"]
    #models = functions
    data_sets = ["digits", "iris"]

    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8

    run_configs = []
    for model in models:
        if isinstance(model, NamedFunction):
            run_configs.append({"model": model, "data_set_name": None})
        else:
            hyperparams = get_hyperparams(model_name=model)
            hp_comb = combinations(hyperparams, 2)
            for hp_conf in hp_comb:
                for ds in data_sets:
                    run_configs.append({"model": model, hp_conf[0]: True, hp_conf[1]: True, "data_set_name": ds})

    logger = get_logger(filename=f"learning_curves/runs_symb/{symb_dir_name}/surr/symb_log_surr_scores.log")

    for run_conf in run_configs:

        if run_conf['data_set_name']:
            data_set = run_conf['data_set_name']
            data_set_postfix = f"_{run_conf['data_set_name']}"
        else:
            data_set = None
            data_set_postfix = ""
        model = run_conf.pop("model")
        if isinstance(model, NamedFunction):
            classifier = model
        else:
            classifier = get_classifier_from_run_conf(model_name=model, run_conf=run_conf)

        function_name = classifier.name if isinstance(classifier, NamedFunction) else model
        optimized_parameters = classifier.configspace.get_hyperparameters()
        parameter_names = [param.name for param in optimized_parameters]

        run_name = f"{function_name.replace(' ', '_')}_{'_'.join(parameter_names)}{data_set_postfix}"

        sampling_dir_surr = f"learning_curves/runs_sampling/surr/{run_name}"
        sampling_dir_smac = f"learning_curves/runs_sampling/smac/{run_name}"
        symb_dir_surr = f"learning_curves/runs_symb/{symb_dir_name}/surr/{run_name}/symb_models"

        with open(f"{sampling_dir_smac}/classifier.pkl", "rb") as classifier_file:
            classifier = pickle.load(classifier_file)
        if isinstance(classifier, NamedFunction):
            classifier_name = classifier.name
        else:
            classifier_name = classifier.name
        optimized_parameters = classifier.configspace.get_hyperparameters()
        parameter_names = [param.name for param in optimized_parameters]

        # Load test data
        logger.info(f"Get test data for {run_name}.")
        try:
            X_test = np.array(
                pd.read_csv(f"learning_curves/runs_symb/{symb_dir_name}/surr/{run_name}/x_test.csv", header=0))
            y_test = np.array(
                pd.read_csv(f"learning_curves/runs_symb/{symb_dir_name}/surr/{run_name}/y_test.csv", header=None))
        except:
            logger.info(f"No test data found, create test data for {run_name}.")
            X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, n_test_samples)
        X_test_reshaped = X_test
        y_test_reshaped = y_test.reshape(-1)

        df_all_metrics = pd.DataFrame()

        for n_samples in N_SAMPLES_SPACING:
            # For smac, get specific sampling file for each sample size for which the number of initial designs differs from
            # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
            if init_design_max_ratio * n_samples < len(
                    optimized_parameters) * init_design_n_configs_per_hyperparamter:
                n_eval = n_samples
            else:
                n_eval = max(N_SAMPLES_SPACING)
            df_train_samples = pd.read_csv(f"{sampling_dir_surr}/samples_{n_eval}.csv")

            sampling_seeds = df_train_samples.seed.unique()

            for sampling_seed in sampling_seeds:
                X_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")[parameter_names]
                y_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")["cost"]
                X_train = X_train_all_samples
                y_train = y_train_all_samples

                for symb_seed in symb_seeds:
                    with open(
                            f"{symb_dir_surr}/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl",
                            "rb") as symb_model_file_surr:
                        symb_surr = pickle.load(symb_model_file_surr)

                    with open(
                            f"{sampling_dir_smac}/surrogates/n_eval{n_eval}_samples{n_samples}_seed{sampling_seed}.pkl",
                            "rb") as surrogate_file:
                        surrogate_model = pickle.load(surrogate_file)

                    df_metrics = get_scores(
                        get_surrogate_predictions(np.array(X_train), classifier, surrogate_model),
                        symb_surr.predict(X_train),
                        get_surrogate_predictions(X_test_reshaped, classifier, surrogate_model),
                        symb_surr.predict(X_test_reshaped)
                    )
                    df_metrics.insert(0, "n_samples", n_samples)
                    df_metrics.insert(1, "sampling_seed", sampling_seed)
                    df_metrics.insert(2, "symb_seed", symb_seed)
                    df_all_metrics = pd.concat((df_all_metrics, df_metrics))

                    df_all_metrics.to_csv(
                        f"learning_curves/runs_symb/{symb_dir_name}/surr/{run_name}/error_metrics_compare_surr.csv",
                        index=False)
