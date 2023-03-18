import openml
import numpy as np
import ConfigSpace as CS
from copy import deepcopy
from typing import Union, Dict
from itertools import combinations
from sklearn.neural_network import MLPClassifier

from hpobench.benchmarks.ml.lr_benchmark import LRBenchmarkBB
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmarkBB
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmarkBB
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmarkBB


ALL_TASKS = openml.tasks.list_tasks()


def get_benchmark_dict():
    benchmark_dict = {
        #LRBenchmarkBB: "LR",
        #SVMBenchmarkBB: "SVM",
        #RandomForestBenchmarkBB: "RF",
        #XGBoostBenchmarkBB: "XGBoost",
        NNBenchmarkBBDefaultHP: "NN",
    }
    return benchmark_dict


def get_task_dict():
    task_ids = [10101, 53, 146818, 146821, 9952, 146822, 31, 3917]
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


class NNBenchmarkBBDefaultHP(NNBenchmarkBB):
    def __init__(self, hyperparameters=None, *args, **kwargs):
        super(NNBenchmarkBBDefaultHP, self).__init__(*args, **kwargs)
        self.configuration_space = self.get_configuration_space(self.rng.randint(0, 10000),
                                                                hyperparameters=hyperparameters)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None,
                                hyperparameters: Union[list, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        if hyperparameters is None or "depth" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    'depth', default_value=3, lower=1, upper=3, log=False)
                )
        if hyperparameters is None or "width" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                'width', default_value=64, lower=16, upper=1024, log=True)
            )
        if hyperparameters is None or "batch_size" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    'batch_size', lower=4, upper=256, default_value=32, log=True)
                )
        if hyperparameters is None or "alpha" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    'alpha', lower=10**-8, upper=1, default_value=10**-3, log=True)
                )
        if hyperparameters is None or "learning_rate_init" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    'learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True)
                )
        return cs

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        config = deepcopy(config)
        if "depth" in config:
            depth = config["depth"]
            config.pop("depth")
        else:
            depth = 3
        if "width" in config:
            width = config["width"]
            config.pop("width")
        else:
            width = 64
        hidden_layers = [width] * depth
        model = MLPClassifier(
            batch_size=config["batch_size"] if "batch_size" in config else 32,
            alpha=config["alpha"] if "alpha" in config else 10**-3,
            learning_rate_init=config["learning_rate_init"] if "learning_rate_init" in config else 10**-3,
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=fidelity['iter'],  # a fidelity being used during initialization
            random_state=rng
        )
        return model

