import pandas as pd
import os
import shutil
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations

from utils.functions_utils import get_functions2d, NamedFunction
from utils.model_utils import get_hyperparams, get_classifier_from_run_conf
from utils.logging_utils import get_logger


N_SAMPLES_SPACING = np.linspace(20, 200, 10, dtype=int).tolist()


if __name__ == "__main__":
    labelsize = 12
    titlesize=14
    symb_dir_name = "default"
    symb_seeds = [0, 3, 6]
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

    # Set up plot directories
    plot_dir = f"learning_curves/plots/symb_fitness"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

    logger = get_logger(filename=f"{plot_dir}/plot_log.log")

    logger.info(f"Save plots to {plot_dir}.")

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

        sampling_dir_smac = f"learning_curves/runs_sampling/smac/{run_name}"

        classifier_titles = {
            "BDT": "Boosted Decision Tree",
            "DT": "Decision Tree",
            "SVM": "Support Vector Machine",
            "MLP": "Neural Network",
        }
        if classifier.name in classifier_titles.keys():
            classifier_title = classifier_titles[classifier.name]
        else:
            classifier_title = classifier.name

        param0 = f"log({optimized_parameters[0].name})" if optimized_parameters[0].log else optimized_parameters[0].name
        param1 = f"log({optimized_parameters[1].name})" if optimized_parameters[1].log else optimized_parameters[1].name

        logger.info(f"Create plot for {run_name}.")

        for n_samples in [100]:
            # For smac, get specific sampling file for each sample size for which the number of initial designs differs from
            # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
            if init_design_max_ratio * n_samples < len(
                    optimized_parameters) * init_design_n_configs_per_hyperparamter:
                n_eval = n_samples
            else:
                n_eval = max(N_SAMPLES_SPACING)
            df_train_samples = pd.read_csv(f"{sampling_dir_smac}/samples_{n_eval}.csv")
            sampling_seeds = df_train_samples.seed.unique()

            for sampling_type in ["SR (BO)", "SR (BO-GP)"]:

                if sampling_type == "SR (BO)":
                    symb_dir = f"learning_curves/runs_symb/{symb_dir_name}/smac/{run_name}/symb_models"
                elif sampling_type == "SR (Random)":
                    symb_dir = f"learning_curves/runs_symb/{symb_dir_name}/rand/{run_name}/symb_models"
                else:
                    symb_dir = f"learning_curves/runs_symb/{symb_dir_name}/surr/{run_name}/symb_models"

                df_all_fitness = pd.DataFrame()

                for sampling_seed in sampling_seeds:
                    for symb_seed in symb_seeds:
                        with open(
                                f"{symb_dir}/n_samples{n_samples}_sampling_seed{sampling_seed}_symb_seed{symb_seed}.pkl",
                                "rb") as symb_model_file:
                            symb_model = pickle.load(symb_model_file)
                        df_fitness = pd.DataFrame({"fitness": symb_model.run_details_["best_fitness"]})
                        df_fitness = df_fitness.reset_index()
                        df_fitness = df_fitness.rename(columns={"index": "generation"})
                        df_fitness.insert(0, "run", f"sampling_seed{sampling_seed}_symb_seed{symb_seed}")
                        # df_fitness.insert(1, "sampling_seed", sampling_seed)
                        # df_fitness.insert(2, "symb_seed", symb_seed)
                        df_all_fitness = pd.concat((df_all_fitness, df_fitness))

                plt.figure()
                _, ax = plt.subplots(figsize=(8, 5))
                sns.lineplot(data=df_all_fitness, x="generation", y="fitness", hue="run")
                              #linestyles="", capsize=0.2, scale=0.7, dodge=0.4)  # , showfliers=False)
                if data_set:
                    plt.title(f"{classifier_title}, Dataset: {data_set}\nOptimize: {param0}, {param1}",
                              fontsize=titlesize)
                else:
                    plt.title(f"{classifier_title}\nOptimize: {param0}, {param1}", fontsize=titlesize)
                plt.ylabel("MSE", fontsize=titlesize)
                plt.yticks(fontsize=labelsize)
                plt.xlabel("Generation", fontsize=titlesize)
                plt.xticks(fontsize=labelsize)
                plt.legend([], [], frameon=False)
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/{run_name}_{sampling_type.replace(' ', '_')}_{n_samples}.png", dpi=400)
                plt.close()
