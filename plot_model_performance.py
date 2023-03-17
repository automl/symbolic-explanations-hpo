import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations

from utils.functions_utils import get_functions2d, NamedFunction
from utils.model_utils import get_hyperparams, get_classifier_from_run_conf
from utils.logging_utils import get_logger


if __name__ == "__main__":
    n_samples_spacing = np.linspace(10, 20, 3, dtype=int).tolist()
    symb_dir_name = "default"
    functions = get_functions2d()
    models = ["MLP", "SVM", "BDT", "DT"]
    #models = functions
    data_sets = ["digits"]
    use_random_samples = False

    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8
    sampling_dir_name = "runs_sampling"

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
    plot_dir = f"learning_curves/plots/performance_plots"
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

        logger.info(f"Create plot for {run_name}.")

        df_samples_all_experiments = pd.DataFrame()

        for sampling_type in ["Random Sampling", "BO Sampling"]:

            if sampling_type == "BO Sampling":
                run_type = "smac"
            else:
                run_type = "rand"
            sampling_dir = f"learning_curves/{sampling_dir_name}/{run_type}"
            sampling_run_dir = f"{sampling_dir}/{run_name}"

            df_train_samples_all = pd.DataFrame(columns=["n_samples", "seed"])

            for n_samples in reversed(n_samples_spacing):
                # Get specific sampling file for each sample size for which the number of initial designs differs from
                # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
                if run_type == "smac" and init_design_max_ratio * n_samples < len(
                        optimized_parameters) * init_design_n_configs_per_hyperparamter:
                    df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{n_samples}.csv")
                else:
                    df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{max(n_samples_spacing)}.csv")

                df_train_samples_all = df_train_samples_all[df_train_samples_all["n_samples"] > n_samples]
                df_train_samples_all = pd.concat((df_train_samples_all, df_train_samples), axis=0)

            df_samples_spacing = df_train_samples_all[df_train_samples_all["n_samples"].isin(n_samples_spacing)]

            df_samples_spacing.insert(0, "Experiment", f"{sampling_type}")
            df_samples_all_experiments = pd.concat((df_samples_all_experiments, df_samples_spacing))

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

        plt.figure()
        _, ax = plt.subplots(figsize=(8, 5))
        sns.pointplot(data=df_train_samples_spacing, x="n_samples", y="cost", errorbar="sd",
                      linestyles="", capsize=0.2, errwidth=0.7, scale=0.7, dodge=0.2)#, showfliers=False)
        if data_set:
            plt.title(f"{classifier_title}, Dataset: {data_set}\nOptimize: {param0}, {param1}", fontsize=16)
        else:
            plt.title(f"{classifier_title}\nOptimize: {param0}, {param1}", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        #plt.yticks(np.arange(0, 20, 2.0), fontsize=14)
        plt.xlabel("Number of Samples", fontsize=16)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        #plt.ylim(0, 18.5)
        #plt.tight_layout(rect=(0, 0.05, 1, 1))
        # sns.move_legend(
        #     ax, "lower center",
        #     bbox_to_anchor=(0.45, -0.32),
        #     ncol=2,
        #     title=None, frameon=False,
        #     fontsize=14
        # )
        plt.savefig(f"{plot_dir}/{run_name}_cost_pointplot.png", dpi=400)
        plt.close()