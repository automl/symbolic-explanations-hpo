import os
import logging
import dill as pickle
import argparse
import sys
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor

from utils.utils import write_dict_to_cfg_file, get_hpo_test_data, get_scores, convert_symb
from utils.symb_reg_utils import get_function_set
from utils import functions

sys.modules['functions'] = functions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')
    args = parser.parse_args()
    job_id = args.job_id

    n_test_samples = 100
    n_seeds = 3
    symb_reg = True
    symb_dir_postfix = "wkendall"
    sampling_run_names = [
        # "rand_BDT_learning_rate_n_estimators_digits_20230221_114624",
        # "rand_BDT_learning_rate_n_estimators_iris_20230221_114624",
        # "rand_Branin_2D_X0_X1_20230221_120527",
        # "rand_Camelback_2D_X0_X1_20230221_120527",
        "rand_DT_max_depth_min_samples_leaf_digits_20230221_114653",
        "rand_DT_max_depth_min_samples_leaf_iris_20230221_114651",
        # "rand_Exponential_function_2D_X0_X1_20230221_120528",
        # "rand_Linear_2D_X0_X1_20230221_120527",
        # "rand_MLP_learning_rate_init_max_iter_digits_20230221_114330",
        # "rand_MLP_learning_rate_init_max_iter_iris_20230221_114330",
        # "rand_MLP_learning_rate_init_n_layer_digits_20230221_114329",
        # "rand_MLP_learning_rate_init_n_layer_iris_20230221_114329",
        # "rand_MLP_learning_rate_init_n_neurons_digits_20230221_114330",
        # "rand_MLP_learning_rate_init_n_neurons_iris_20230221_114330",
        # "rand_MLP_max_iter_n_layer_digits_20230221_114332",
        # "rand_MLP_max_iter_n_layer_iris_20230221_114330",
        # "rand_MLP_max_iter_n_neurons_digits_20230221_114332",
        # "rand_MLP_max_iter_n_neurons_iris_20230221_114330",
        # "rand_MLP_n_layer_n_neurons_digits_20230221_114330",
        # "rand_MLP_n_layer_n_neurons_iris_20230221_114330",
        # "rand_Polynom_function_2D_X0_X1_20230221_120528",
        # "rand_Rosenbrock_2D_X0_X1_20230221_120527",
        # "rand_SVM_C_coef0_digits_20230221_114754",
        # "rand_SVM_C_coef0_iris_20230221_114755",
        # "rand_SVM_C_degree_digits_20230221_114756",
        # "rand_SVM_C_degree_iris_20230221_114756",
        # "rand_SVM_C_gamma_digits_20230221_114754",
        # "rand_SVM_C_gamma_iris_20230221_114755",
        # "rand_SVM_coef0_degree_digits_20230221_114754",
        # "rand_SVM_coef0_degree_iris_20230221_114756",
        # "rand_SVM_coef0_gamma_digits_20230221_114755",
        # "rand_SVM_coef0_gamma_iris_20230221_114755",
        # "rand_SVM_degree_gamma_digits_20230221_114754",
        # "rand_SVM_degree_gamma_iris_20230221_114754",
        # "smac_BDT_learning_rate_n_estimators_digits_20230223_162320",
        # "smac_BDT_learning_rate_n_estimators_iris_20230223_162320",
        # "smac_Branin_2D_X0_X1_20230223_162155",
        # "smac_Camelback_2D_X0_X1_20230223_162155",
        "smac_DT_max_depth_min_samples_leaf_digits_20230224_090309",
        "smac_DT_max_depth_min_samples_leaf_iris_20230224_090310",
        #"smac_Exponential_function_2D_X0_X1_20230224_144857",
        # "smac_Linear_2D_X0_X1_20230223_162155",
        # "smac_MLP_learning_rate_init_max_iter_digits_20230223_162437",
        # "smac_MLP_learning_rate_init_max_iter_iris_20230223_162436",
        # "smac_MLP_learning_rate_init_n_layer_digits_20230223_162436",
        # "smac_MLP_learning_rate_init_n_layer_iris_20230223_162436",
        # "smac_MLP_learning_rate_init_n_neurons_digits_20230223_162436",
        # "smac_MLP_learning_rate_init_n_neurons_iris_20230223_162436",
        # "smac_MLP_max_iter_n_layer_digits_20230223_162436",
        # "smac_MLP_max_iter_n_layer_iris_20230223_162436",
        # "smac_MLP_max_iter_n_neurons_digits_20230223_162436",
        # "smac_MLP_max_iter_n_neurons_iris_20230223_162436",
        # "smac_MLP_n_layer_n_neurons_digits_20230223_162436",
        # "smac_MLP_n_layer_n_neurons_iris_20230223_162437",
        #"smac_Polynom_function_2D_X0_X1_20230227_165032/",
        # "smac_Rosenbrock_2D_X0_X1_20230223_162155",
        # "smac_SVM_C_coef0_digits_20230223_164415",
        # "smac_SVM_C_coef0_iris_20230223_162859",
        # "smac_SVM_C_degree_digits_20230223_162900",
        # "smac_SVM_C_degree_iris_20230223_164415",
        # "smac_SVM_C_gamma_digits_20230223_162900",
        # "smac_SVM_C_gamma_iris_20230223_162859",
        # "smac_SVM_coef0_degree_digits_20230223_162859",
        # "smac_SVM_coef0_degree_iris_20230223_162859",
        # "smac_SVM_coef0_gamma_digits_20230223_162859",
        # "smac_SVM_coef0_gamma_iris_20230223_162859",
        # "smac_SVM_degree_gamma_digits_20230223_162900",
        # "smac_SVM_degree_gamma_iris_20230223_162859"
    ]
    sampling_run_name = sampling_run_names[int(job_id)]

    run_dir = f"learning_curves/runs/{sampling_run_name}"
    sampling_dir = f"{run_dir}/sampling"
    symb_dir = f"{run_dir}/symb_{symb_dir_postfix}"
    if not os.path.exists(f"{symb_dir}/symb_models"):
        os.makedirs(f"{symb_dir}/symb_models")
    model = sampling_run_name.split("_")[1]

    # setup logging
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(filename=f"{symb_dir}/symb_log.log", encoding="utf8")
    handler.setLevel("INFO")
    handler.setFormatter(
        logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    logger.root.addHandler(handler)
    handler2 = logging.StreamHandler()
    handler2.setLevel("INFO")
    handler2.setFormatter(
        logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    handler2.setStream(sys.stdout)
    logger.root.addHandler(handler2)
    logger.root.setLevel("INFO")

    with open(f"{sampling_dir}/classifier.pkl", "rb") as classifier_file:
        classifier = pickle.load(classifier_file)

    optimized_parameters = classifier.configspace.get_hyperparameters()
    param_names = [param.name for param in optimized_parameters]

    logger.info(f"Fit Symbolic Model for {sampling_run_name}.")

    X_test, y_test = get_hpo_test_data(classifier, optimized_parameters, n_test_samples)

    df_train_samples = pd.read_csv(f"{sampling_dir}/samples.csv")
    sampling_seeds = df_train_samples.seed.unique()

    n_samples_spacing = np.linspace(20, 200, 10)

    df_all_metrics = pd.DataFrame()
    df_all_complexity = pd.DataFrame()
    df_all_expr = pd.DataFrame()

    symb_params = dict(
        population_size=5000,
        generations=50,
        #stopping_criteria=0.0001,
        #p_crossover=0.7,
        #p_subtree_mutation=0.1,
        #p_hoist_mutation=0.05,
        #p_point_mutation=0.1,
        #max_samples=0.9,
        #parsimony_coefficient=0.01,
        function_set=get_function_set(),
        metric="mse",  # "mean absolute error",
        verbose=1,
        #const_range=(
        #    -100,
        #    100,
        #),  # Range for constants, rather arbitrary setting here?
    )

    write_dict_to_cfg_file(
        dictionary=symb_params,
        target_file_path=f"{symb_dir}/symbolic_regression_params.cfg",
    )

    for sampling_seed in sampling_seeds:
        X_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")[param_names]
        y_train_all_samples = df_train_samples.query(f"seed == {sampling_seed}")["cost"]

        for n_samples in n_samples_spacing.astype(int):

            X_train = X_train_all_samples[:n_samples]
            y_train = y_train_all_samples[:n_samples]

            logger.info(f"Fit Symbolic Model for {n_samples} samples and sampling seed {sampling_seed}.")

            if symb_reg:
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
                        y_test.reshape(-1),
                        symb_model.predict(X_test.reshape(len(optimized_parameters), -1).T)
                    )
                    df_metrics.insert(0, "n_samples", n_samples)
                    df_metrics.insert(0, "sampling_seed", sampling_seed)
                    df_metrics.insert(0, "symb_seed", symb_seed)
                    df_all_metrics = pd.concat((df_all_metrics, df_metrics))

                    complexity = symb_model._program.length_
                    df_complexity = pd.DataFrame({
                        "complexity": [complexity],
                        "n_samples": [n_samples],
                        "sampling_seed": [sampling_seed],
                        "symb_seed": [symb_seed]
                    })
                    df_all_complexity = pd.concat((df_all_complexity, df_complexity))

                    try:
                        df_expr = pd.DataFrame(
                            {"expr": [convert_symb(symb_model, n_dim=len(optimized_parameters), n_decimals=3)]})
                    except:
                        df_expr = pd.DataFrame({"expr": [""]})
                        print(f"Could not convert expression for n_samples: {n_samples}, sampling_seed: {sampling_seed}"
                              f", symb_seed: {symb_seed}.")
                    df_expr.insert(0, "n_samples", n_samples)
                    df_expr.insert(0, "sampling_seed", sampling_seed)
                    df_expr.insert(0, "symb_seed", symb_seed)
                    df_all_expr = pd.concat((df_all_expr, df_expr))

                    df_all_metrics.to_csv(f"{symb_dir}/error_metrics.csv", index=False)
                    df_all_complexity.to_csv(f"{symb_dir}/complexity.csv", index=False)
                    df_all_expr.to_csv(f"{symb_dir}/expressions.csv", index=False)
