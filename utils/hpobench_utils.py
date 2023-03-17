import openml
from itertools import combinations

from hpobench.benchmarks.ml.lr_benchmark import LRBenchmarkBB
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmarkBB
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmarkBB
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmarkBB


ALL_TASKS = openml.tasks.list_tasks()


def get_benchmark_dict():
    benchmark_dict = {
        LRBenchmarkBB: "LR",
        SVMBenchmarkBB: "SVM",
        RandomForestBenchmarkBB: "RF",
        XGBoostBenchmarkBB: "XGBoost",
        NNBenchmarkBB: "NN",
    }
    return benchmark_dict


def get_task_dict():
    task_ids = [10101] #, 53, 146818, 146821, 9952, 146822, 31, 3917]
    task_dict = {tid: ALL_TASKS[tid]["name"] for tid in task_ids}
    return task_dict


def get_run_config(job_id, n_optimized_params):
    run_configs = []
    for benchmark in get_benchmark_dict().keys():
        hyperparams = benchmark.get_configuration_space().get_hyperparameter_names()
        hp_comb = combinations(hyperparams, n_optimized_params)
        for hp_conf in hp_comb:
            for task_id in get_task_dict().keys():
                run_configs.append({"benchmark": benchmark, "task_id": task_id, "hp_conf": hp_conf})
    return run_configs[int(job_id)]
