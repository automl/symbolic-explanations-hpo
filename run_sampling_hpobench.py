import os
import numpy as np
import pandas as pd
import shutil
import argparse
import dill as pickle
from smac import BlackBoxFacade, Callback

from utils.logging_utils import get_logger
from utils.run_utils import get_surrogate_predictions
from utils.smac_utils import run_smac_optimization
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict


class SurrogateModelCallback(Callback):
    def on_next_configurations_end(self, config, config_selector):
        if config._acquisition_function._eta:
            surrogate_model = config._model
            processed_configs = len(config._processed_configs)
            if processed_configs in n_samples_spacing:
                with open(
                        f"{sampling_run_dir}/surrogates/n_eval{n_samples}_samples{processed_configs}_seed{seed}.pkl",
                        "wb") as surrogate_file:
                    pickle.dump(surrogate_model, surrogate_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    parser.add_argument('--run_type',
                        choices=["smac", "rand", "surr"],
                        help=
                        '"smac": Collect samples via Bayesian optimization, '
                        '"rand": Collect randomly sampled configurations and evaluate their performance,'
                        '"surr" Collect random samples, but estimated their performance using the Gaussian process '
                        '(Please note that, in the latter case, the BO sampling needs to be run beforehand to '
                        'provide the Gaussian process models.)'
                        )
    args = parser.parse_args()
    run_type = args.run_type

    # number of HPs to optimize
    n_optimized_params = 5
    # number of HP combinations to consider per model
    max_hp_comb = 1

    n_samples_spacing = np.linspace(200, 200, 1, dtype=int).tolist()
    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8

    sampling_dir_name = "runs_sampling_hpobench_new"
    n_seeds = 2
    surrogate_n_samples = 400

    run_conf = get_run_config(job_id=args.job_id, n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)

    task_dict = get_task_dict()
    data_set_postfix = f"_{task_dict[run_conf['task_id']]}"
    optimized_parameters = list(run_conf["hp_conf"])
    model_name = get_benchmark_dict()[run_conf["benchmark"]]
    b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

    def optimization_function_wrapper(cfg, seed):
        """ Helper-function: simple wrapper to use the benchmark with smac """
        result_dict = b.objective_function(cfg, rng=seed)
        return result_dict['function_value']

    if run_type == "rand":
        n_samples_to_eval = [max(n_samples_spacing)]
    else:
        # SMAC uses at most scenario.n_trials * max_ratio number of configurations in the initial design
        # If we run SMAC only once with n_trials = max(n_samples_spacing), we would always use the maximum number of
        # initial designs, e.g. a run with 20 samples would have the same number of initial designs as a run with 200
        # Thus, we do separate runs for each sample size as long as the number of initial designs would differ
        if run_type == "surr":
            n_samples_to_eval = n_samples_spacing
        else:
            n_samples_to_eval = [n for n in n_samples_spacing if
                                 init_design_max_ratio * n < len(optimized_parameters) * init_design_n_configs_per_hyperparamter]
            if max(n_samples_spacing) not in n_samples_to_eval:
                n_samples_to_eval.append(max(n_samples_spacing))

    run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}{data_set_postfix}"

    sampling_dir = f"results/{sampling_dir_name}/{run_type}"
    sampling_run_dir = f"{sampling_dir}/{run_name}"
    if os.path.exists(sampling_run_dir):
        shutil.rmtree(sampling_run_dir)
    os.makedirs(sampling_run_dir)
    if run_type == "smac":
        os.makedirs(f"{sampling_run_dir}/surrogates")

    # setup logging
    logger = get_logger(filename=f"{sampling_run_dir}/sampling_log.log")

    logger.info(f"Start {run_type} sampling for {run_name}.")

    for n_samples in n_samples_to_eval:

        logger.info(f"Run: {run_name}")
        logger.info(f"Start run to sample {n_samples} samples.")

        # required for surrogate evaluation
        if init_design_max_ratio * n_samples < len(optimized_parameters) * init_design_n_configs_per_hyperparamter:
            n_eval = n_samples
        else:
            n_eval = max(n_samples_spacing)

        df_samples = pd.DataFrame()

        for i in range(n_seeds):
            seed = i * 3

            np.random.seed(seed)

            # add only parameters to be optimized to configspace
            cs = b.get_configuration_space(seed=seed, hyperparameters=optimized_parameters)

            logger.info(f"Run: {run_name}")
            logger.info(f"Sample configs and train {model_name} with seed {seed}.")

            if run_type == "rand":
                configurations = cs.sample_configuration(size=n_samples)
                performances = np.array(
                    [b.objective_function(config.get_dictionary(), seed=seed)["function_value"] for config in
                     configurations]
                )
                configurations = np.array(
                    [list(i.get_dictionary().values()) for i in configurations]
                ).T
            elif run_type == "surr":
                configurations = cs.sample_configuration(size=surrogate_n_samples)
                configurations = np.array(
                    [list(i.get_dictionary().values()) for i in configurations]
                ).T
                with open(f"results/{sampling_dir_name}/smac/{run_name}/surrogates/n_eval{n_eval}"
                          f"_samples{n_samples}_seed{seed}.pkl", "rb") as surrogate_file:
                    surrogate_model = pickle.load(surrogate_file)
                performances = np.array(get_surrogate_predictions(configurations.T, cs, surrogate_model))
            else:
                configurations, performances, _ = run_smac_optimization(
                    configspace=cs,
                    facade=BlackBoxFacade,
                    target_function=optimization_function_wrapper,
                    function_name=model_name,
                    n_eval=n_samples,
                    run_dir=sampling_run_dir,
                    seed=seed,
                    n_configs_per_hyperparamter=init_design_n_configs_per_hyperparamter,
                    max_ratio=init_design_max_ratio,
                    callback=SurrogateModelCallback()
                )

            df = pd.DataFrame(
                data=np.concatenate((configurations.T,
                                     performances.reshape(-1, 1)), axis=1),
                columns=optimized_parameters + ["cost"])
            df.insert(0, "seed", seed)
            df = df.reset_index()
            df = df.rename(columns={"index": "n_samples"})
            df["n_samples"] = df["n_samples"] + 1
            df_samples = pd.concat((df_samples, df))

            df_samples.to_csv(f"{sampling_run_dir}/samples_{n_samples}.csv", index=False)

