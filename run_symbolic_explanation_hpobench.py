import os
import dill as pickle
import argparse
import numpy as np
import pandas as pd
import sympy
import shutil
from gplearn.genetic import SymbolicRegressor

from utils.logging_utils import get_logger
from utils.run_utils import write_dict_to_cfg_file, get_hpo_test_data, get_scores, convert_symb
from utils.symb_reg_utils import get_function_set
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    args = parser.parse_args()

    use_random_samples = False
    evaluate_on_surrogate = False
    symb_dir_name = "parsimony0005"
    sampling_dir_name = "runs_sampling_hpobench"
    dir_with_test_data = "learning_curves/runs_surr_hpobench"
    n_optimized_params = 2
    n_samples_spacing = np.linspace(20, 200, 10, dtype=int).tolist()
    n_seeds = 3
    n_test_samples = 100
    init_design_max_ratio = 0.25
    init_design_n_configs_per_hyperparamter = 8

    run_conf = get_run_config(job_id=args.job_id, n_optimized_params=n_optimized_params)

    task_dict = get_task_dict()
    data_set_postfix = f"_{task_dict[run_conf['task_id']]}"
    optimized_parameters = list(run_conf["hp_conf"])
    model_name = get_benchmark_dict()[run_conf["benchmark"]]
    b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

    # add only parameters to be optimized to configspace
    cs = b.get_configuration_space(seed=0, hyperparameters=optimized_parameters)

    if use_random_samples:
        run_type = "rand"
    elif evaluate_on_surrogate:
        run_type = "surr"
    else:
        run_type = "smac"

    run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}{data_set_postfix}"

    sampling_dir = f"learning_curves/{sampling_dir_name}/{run_type}"
    sampling_run_dir = f"{sampling_dir}/{run_name}"

    symb_dir = f"learning_curves/runs_symb_hpobench/{symb_dir_name}/{run_type}/{run_name}"
    if os.path.exists(symb_dir):
        shutil.rmtree(symb_dir)
    os.makedirs(f"{symb_dir}/symb_models")

    logger = get_logger(filename=f"{symb_dir}/symb_log.log")

    logger.info(f"Fit Symbolic Model for {run_name} ({run_type}).")

    logger.info(f"Get and save test data.")
    if dir_with_test_data:
        X_test = get_hpo_test_data(b, cs.get_hyperparameters(), n_test_samples, return_x=True)
        y_test = np.array(
        pd.read_csv(f"{dir_with_test_data}/{run_name}/y_test.csv", header=None))
        y_test = y_test.reshape(X_test.shape[1], X_test.shape[2])
    else:
        logger.info(f"No previous test data dir provided, create test data for {run_name}.")
        X_test, y_test = get_hpo_test_data(b, cs.get_hyperparameters(), n_test_samples)
    X_test_reshaped = X_test.reshape(len(optimized_parameters), -1).T
    y_test_reshaped = y_test.reshape(-1)
    pd.DataFrame(X_test_reshaped, columns=optimized_parameters).to_csv(f"{symb_dir}/x_test.csv", index=False)
    pd.DataFrame(y_test_reshaped).to_csv(f"{symb_dir}/y_test.csv", header=False, index=False)

    df_all_metrics = pd.DataFrame()
    df_all_complexity = pd.DataFrame()
    df_all_expr = pd.DataFrame()

    symb_params = dict(
        population_size=5000,
        generations=20,
        function_set=get_function_set(),
        metric="rmse",
        parsimony_coefficient=0.0005,
        verbose=1,
    )

    write_dict_to_cfg_file(
        dictionary=symb_params,
        target_file_path=f"{symb_dir}/symbolic_regression_params.cfg",
    )

    for n_samples in n_samples_spacing:
        # For smac, get specific sampling file for each sample size for which the number of initial designs differs from
        # the maximum number of initial designs (number of hyperparameters * init_design_n_configs_per_hyperparamter)
        if run_type == "surr" or (run_type == "smac" and init_design_max_ratio * n_samples < len(
                optimized_parameters) * init_design_n_configs_per_hyperparamter):
            df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{n_samples}.csv")
        else:
            df_train_samples = pd.read_csv(f"{sampling_run_dir}/samples_{max(n_samples_spacing)}.csv")

        sampling_seeds = df_train_samples.seed.unique()

        for sampling_seed in sampling_seeds:
            X_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")[optimized_parameters]
            y_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")["cost"]


            if evaluate_on_surrogate:
                X_train = X_train_all_samples
                y_train = y_train_all_samples
            else:
                X_train = X_train_all_samples[:n_samples]
                y_train = y_train_all_samples[:n_samples]

            if len(X_train) < n_samples:
                logger.warning(
                    f"Found less than {n_samples} when trying to evaluate {n_samples} samples for sampling seed "
                    f"{sampling_seed}, skip.")
                break

            logger.info(f"Fit Symbolic Model for {n_samples} samples and sampling seed {sampling_seed}.")

            for i in range(n_seeds):
                symb_seed = i * 3

                logger.info(f"Using seed {symb_seed} for symbolic regression.")

                # run SR on SMAC samples
                symb_model = SymbolicRegressor(**symb_params, random_state=symb_seed)
                symb_model.fit(X_train, y_train)

                # pickle symbolic regression model
                with open(
                        f"{symb_dir}/symb_models/n_samples{n_samples}_sampling_seed{sampling_seed}_"
                        f"symb_seed{symb_seed}.pkl", "wb") as symb_model_file:
                    # pickling all programs lead to huge files
                    delattr(symb_model, "_programs")
                    pickle.dump(symb_model, symb_model_file)

                df_metrics = get_scores(
                    y_train,
                    symb_model.predict(X_train),
                    y_test_reshaped,
                    symb_model.predict(X_test_reshaped)
                )
                df_metrics.insert(0, "n_samples", n_samples)
                df_metrics.insert(1, "sampling_seed", sampling_seed)
                df_metrics.insert(2, "symb_seed", symb_seed)
                df_all_metrics = pd.concat((df_all_metrics, df_metrics))


                program_length_before_simplification = symb_model._program.length_
                try:
                    conv_expr = convert_symb(symb_model, n_dim=len(optimized_parameters), n_decimals=3)
                except:
                    conv_expr = ""
                    logger.warning(f"Could not convert expression for n_samples: {n_samples}, "
                                   f"sampling_seed: {sampling_seed}, symb_seed: {symb_seed}.")
                try:
                    program_operations = sympy.count_ops(conv_expr)
                except:
                    try:
                        program_operations = sympy.count_ops(symb_model)
                    except:
                        program_operations = -1

                df_expr = pd.DataFrame({"expr": [conv_expr]})
                df_expr.insert(0, "n_samples", n_samples)
                df_expr.insert(1, "sampling_seed", sampling_seed)
                df_expr.insert(2, "symb_seed", symb_seed)
                df_all_expr = pd.concat((df_all_expr, df_expr))

                df_complexity = pd.DataFrame({
                    "n_samples": [n_samples],
                    "sampling_seed": [sampling_seed],
                    "symb_seed": [symb_seed],
                    "program_operations": [program_operations],
                    "program_length_before_simplification": [program_length_before_simplification],
                })
                df_all_complexity = pd.concat((df_all_complexity, df_complexity))

                df_all_metrics.to_csv(f"{symb_dir}/error_metrics.csv", index=False)
                df_all_complexity.to_csv(f"{symb_dir}/complexity.csv", index=False)
                df_all_expr.to_csv(f"{symb_dir}/expressions.csv", index=False)
