import pandas as pd
import dill as pickle
import sympy
from utils.run_utils import convert_symb

from logging import getLogger
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict

if __name__ == "__main__":
    symb_dir_name = "parsimony0005"
    n_optimized_params = 2

    run_configs = get_run_config(n_optimized_params=n_optimized_params)

    logger = getLogger(__name__)

    logger.info(f"Run simplification correction for symb dir {symb_dir_name}.")

    for run_conf in run_configs[0]:

        task_dict = get_task_dict()
        data_set = f"{task_dict[run_conf['task_id']]}"
        optimized_parameters = list(run_conf["hp_conf"])
        model_name = get_benchmark_dict()[run_conf["benchmark"]]
        b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

        # add only parameters to be optimized to configspace
        cs = b.get_configuration_space(hyperparameters=optimized_parameters)

        run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}_{data_set}"

        logger.info(f"Correct for {run_name}.")

        for sampling_type in ["GP Baseline", "SR (BO)", "SR (Random)"]:
            if sampling_type == "SR (BO)":
                symb_dir = f"learning_curves/runs_symb_hpobench/{symb_dir_name}/smac/{run_name}"
            elif sampling_type == "SR (Random)":
                symb_dir = f"learning_curves/runs_symb_hpobench/{symb_dir_name}/rand/{run_name}"
            else:
                symb_dir = f"learning_curves/runs_symb_hpobench/{symb_dir_name}/surr/{run_name}"

            df_expr_old = pd.read_csv(f"{symb_dir}/expressions.csv")
            df_complexity_old = pd.read_csv(f"{symb_dir}/complexity.csv")

            df_expr_old.to_csv(f"{symb_dir}/expressions_old.csv", index=False)
            df_complexity_old.to_csv(f"{symb_dir}/complexity_old.csv", index=False)

            df_all_expr = pd.DataFrame()
            df_all_complexity = pd.DataFrame()

            for index, row in df_expr_old.iterrows():
                with open(
                        f"{symb_dir}/symb_models/n_samples{row['n_samples']}_sampling_seed{row['sampling_seed']}_symb_seed{row['symb_seed']}.pkl",
                        "rb") as symb_model_file:
                    symb_model = pickle.load(symb_model_file)

                program_length_before_simplification = symb_model._program.length_
                try:
                    conv_expr = convert_symb(symb_model, n_dim=len(optimized_parameters), n_decimals=3)
                except:
                    conv_expr = ""
                    logger.warning(f"Could not convert expression for n_samples: {row['n_samples']}, "
                                   f"sampling_seed: {row['sampling_seed']}, symb_seed: {row['symb_seed']}.")
                try:
                    program_operations = sympy.count_ops(conv_expr)
                except:
                    try:
                        program_operations = sympy.count_ops(symb_model)
                    except:
                        program_operations = -1

                df_expr = pd.DataFrame({"expr_simplified": [conv_expr], "expr": symb_model._program})
                df_expr.insert(0, "n_samples", row['n_samples'])
                df_expr.insert(1, "sampling_seed", row['sampling_seed'])
                df_expr.insert(2, "symb_seed", row['symb_seed'])
                df_all_expr = pd.concat((df_all_expr, df_expr))

                df_complexity = pd.DataFrame({
                    "n_samples": [row['n_samples']],
                    "sampling_seed": [row['sampling_seed']],
                    "symb_seed": [row['symb_seed']],
                    "program_operations": [program_operations],
                    "program_length_before_simplification": [program_length_before_simplification],
                })
                df_all_complexity = pd.concat((df_all_complexity, df_complexity))

                df_all_expr.to_csv(f"{symb_dir}/expressions.csv", index=False)
                df_all_complexity.to_csv(f"{symb_dir}/complexity.csv", index=False)
