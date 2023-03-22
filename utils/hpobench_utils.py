import openml
import numpy as np
import ConfigSpace as CS
from copy import deepcopy
from typing import Union, Dict
from itertools import combinations

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from hpobench.benchmarks.ml.lr_benchmark import LRBenchmarkBB
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmarkBB
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmarkBB
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmarkBB


ALL_TASKS = openml.tasks.list_tasks()


def get_benchmark_dict():
    benchmark_dict = {
        LRBenchmarkBBDefaultHP: "LR",
        SVMBenchmarkBBDefaultHP: "SVM",
        RandomForestBenchmarkBBDefaultHP: "RF",
        XGBoostBenchmarkBBDefaultHP: "XGBoost",
        # NNBenchmarkBBDefaultHP: "NN",
    }
    return benchmark_dict


def get_task_dict():
    task_ids = [10101, 53, 146818, 146821, 9952, 146822, 31, 3917] #eval NN only on those 8 datasets
    #task_ids = [10101, 53]
    #task_ids = [146818, 31]
    #task_ids = [168912, 3, 167119, 12, 146212, 168911, 9981, 167120, 14965, 146606, 7592, 9977] # 12 datasets
    task_dict = {tid: ALL_TASKS[tid]["name"] for tid in task_ids}
    return task_dict


def get_run_config(n_optimized_params, max_hp_comb=None, job_id=None, parsimony_coefficient_space=None):
    run_configs = []
    for benchmark in get_benchmark_dict().keys():
        hyperparams = benchmark.get_configuration_space().get_hyperparameter_names()
        hp_comb = combinations(hyperparams, n_optimized_params)
        if max_hp_comb:
            hpc = list(hp_comb)[:max_hp_comb]
        else:
            hpc = hp_comb
        for hp_conf in hpc:
            for task_id in get_task_dict().keys():
                if parsimony_coefficient_space:
                    for parsimony in parsimony_coefficient_space:
                        run_configs.append(
                            {"benchmark": benchmark, "task_id": task_id, "hp_conf": hp_conf, "parsimony": parsimony})
                else:
                    run_configs.append({"benchmark": benchmark, "task_id": task_id, "hp_conf": hp_conf})
    if job_id:
        return run_configs[int(job_id)]
    else:
        return run_configs


class LRBenchmarkBBDefaultHP(LRBenchmarkBB):
    def __init__(self, hyperparameters=None, *args, **kwargs):
        super(LRBenchmarkBBDefaultHP, self).__init__(*args, **kwargs)
        self.configuration_space = self.get_configuration_space(self.rng.randint(0, 10000),
                                                                hyperparameters=hyperparameters)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None,
                                hyperparameters: Union[list, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        if hyperparameters is None or "alpha" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    "alpha", 1e-5, 1, log=True, default_value=1e-3)
            ),
        if hyperparameters is None or "eta0" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    "eta0", 1e-5, 1, log=True, default_value=1e-2)
            )
        return cs

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        # initializing model
        rng = self.rng if rng is None else rng

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        # https://scikit-learn.org/stable/modules/sgd.html
        model = SGDClassifier(
            alpha=config["alpha"] if "alpha" in config else 1e-3,
            eta0=config["eta0"] if "eta0" in config else 1e-2,
            loss="log",  # performs Logistic Regression
            max_iter=fidelity["iter"],
            learning_rate="adaptive",
            tol=None,
            random_state=rng,

        )
        return model


class SVMBenchmarkBBDefaultHP(SVMBenchmarkBB):
    def __init__(self, hyperparameters=None, *args, **kwargs):
        super(SVMBenchmarkBBDefaultHP, self).__init__(*args, **kwargs)
        self.configuration_space = self.get_configuration_space(self.rng.randint(0, 10000),
                                                                hyperparameters=hyperparameters)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None,
                                hyperparameters: Union[list, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        # https://jmlr.org/papers/volume20/18-444/18-444.pdf (Table 1)
        if hyperparameters is None or "C" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    "C", 2**-10, 2**10, log=True, default_value=1.0)
            ),
        if hyperparameters is None or "gamma" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    "gamma", 2**-10, 2**10, log=True, default_value=0.1)
            )
        return cs

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        # initializing model
        rng = self.rng if rng is None else rng
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        model = SVC(
            C=config["C"] if "C" in config else 1.0,
            gamma=config["gamma"] if "gamma" in config else 0.1,
            random_state=rng,
            cache_size=self.cache_size
        )
        return model


class RandomForestBenchmarkBBDefaultHP(RandomForestBenchmarkBB):
    def __init__(self, hyperparameters=None, *args, **kwargs):
        super(RandomForestBenchmarkBBDefaultHP, self).__init__(*args, **kwargs)
        self.configuration_space = self.get_configuration_space(self.rng.randint(0, 10000),
                                                                hyperparameters=hyperparameters)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None,
                                hyperparameters: Union[list, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        if hyperparameters is None or "max_depth" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    'max_depth', lower=1, upper=50, default_value=10, log=True)
            ),
        if hyperparameters is None or "min_samples_split" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    'min_samples_split', lower=2, upper=128, default_value=32, log=True)
            ),
        if hyperparameters is None or "max_features" in hyperparameters:
            cs.add_hyperparameter(
            # the use of a float max_features is different than the sklearn usage
                CS.UniformFloatHyperparameter(
                    'max_features', lower=0, upper=1.0, default_value=0.5, log=False)
            ),
        if hyperparameters is None or "min_samples_leaf" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    'min_samples_leaf', lower=1, upper=20, default_value=1, log=False)
            ),
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
        n_features = self.train_X.shape[1]
        if "max_features" in config:
            max_features = int(np.rint(np.power(n_features, config["max_features"])))
        else:
            max_features = int(np.rint(np.power(n_features, 0.5)))
        model = RandomForestClassifier(
            max_depth=config["max_depth"] if "max_depth" in config else 10,
            min_samples_split=config["min_samples_split"] if "min_samples_split" in config else 32,
            max_features=max_features,
            min_samples_leaf=config["min_samples_leaf"] if "min_samples_leaf" in config else 1,
            n_estimators=fidelity['n_estimators'],  # a fidelity being used during initialization
            bootstrap=True,
            random_state=rng
        )
        return model


class XGBoostBenchmarkBBDefaultHP(XGBoostBenchmarkBB):
    def __init__(self, hyperparameters=None, *args, **kwargs):
        super(XGBoostBenchmarkBBDefaultHP, self).__init__(*args, **kwargs)
        self.configuration_space = self.get_configuration_space(self.rng.randint(0, 10000),
                                                                hyperparameters=hyperparameters)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None,
                                hyperparameters: Union[list, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        if hyperparameters is None or "eta" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    'eta', lower=2**-10, upper=1., default_value=0.3, log=True)
            ),  # learning rate
        if hyperparameters is None or "max_depth" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    'max_depth', lower=1, upper=50, default_value=10, log=True)
            ),
        if hyperparameters is None or "colsample_bytree" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    'colsample_bytree', lower=0.1, upper=1., default_value=1., log=False)
            ),
        if hyperparameters is None or "reg_lambda" in hyperparameters:
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    'reg_lambda', lower=2**-10, upper=2**10, default_value=1, log=True)
            )
        return cs

    def init_model(self,
                   config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        rng = rng if (rng is None or isinstance(rng, int)) else self.seed
        extra_args = dict(
            booster="gbtree",
            n_estimators=fidelity['n_estimators'],
            objective="binary:logistic",
            random_state=rng,
            subsample=1
        )
        if self.n_classes > 2:
            extra_args["objective"] = "multi:softmax"
            extra_args.update({"num_class": self.n_classes})

        model = xgb.XGBClassifier(
            eta=config["eta"] if "eta" in config else 0.3,
            max_depth=config["max_depth"] if "max_depth" in config else 10,
            colsample_bytree=config["colsample_bytree"] if "colsample_bytree" in config else 1.,
            reg_lambda=config["reg_lambda"] if "reg_lambda" in config else 1,
            **extra_args
        )
        return model


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

