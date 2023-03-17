from itertools import combinations

from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark


def get_benchmarks():
    benchmarks = [LRBenchmark] #, SVMBenchmark, RandomForestBenchmark, XGBoostBenchmark, NNBenchmark]
    return benchmarks


def get_task_ids():
    task_ids = [146818]
    return task_ids


def get_run_config(job_id, n_optimized_params):
    run_configs = []
    for benchmark in get_benchmarks():
        hyperparams = benchmark.get_configuration_space().get_hyperparameter_names()
        hp_comb = combinations(hyperparams, n_optimized_params)
        for hp_conf in hp_comb:
            for task_id in get_task_ids():
                run_configs.append({"benchmark": benchmark, "task_id": task_id, "hp_conf": hp_conf})
    return run_configs[int(job_id)]


def get_model_name(benchmark) -> str:
    if isinstance(benchmark, LRBenchmark):
        return "LR"
    elif isinstance(benchmark, SVMBenchmark):
        return "SVM"
    elif isinstance(benchmark, RandomForestBenchmark):
        return "RF"
    elif isinstance(benchmark, XGBoostBenchmark):
        return "XGBoost"
    elif isinstance(benchmark, NNBenchmark):
        return "NN"
    else:
        raise Exception("Unknown benchmark.")
