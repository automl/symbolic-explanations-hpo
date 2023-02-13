from sklearn import datasets, svm

import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
digits = load_digits()


class MLP:
    def __init__(
        self,
        optimize_n_neurons=False,
        optimize_n_layer=False,
        optimize_activation=False,
        optimize_solver=False,
        optimize_batch_size=False,
        optimize_learning_rate_init=False,
        optimize_max_iter=False,
        seed=0,
    ):
        self.optimize_n_neurons = optimize_n_neurons
        self.optimize_n_layer = optimize_n_layer
        self.optimize_activation = optimize_activation
        self.optimize_solver = optimize_solver
        self.optimize_batch_size = optimize_batch_size
        self.optimize_learning_rate_init = optimize_learning_rate_init
        self.optimize_max_iter = optimize_max_iter
        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical(
            "activation", ["logistic", "tanh", "relu"], default="tanh"
        )
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate_init = Float(
            "learning_rate_init", (0.0001, 1.0), default=0.001, log=True
        )
        max_iter = Integer("max_iter", (10, 100), default=25)

        if self.optimize_n_layer:
            cs.add_hyperparameter(n_layer)
        if self.optimize_n_neurons:
            cs.add_hyperparameter(n_neurons)
        if self.optimize_activation:
            cs.add_hyperparameter(activation)
        if self.optimize_solver:
            cs.add_hyperparameter(solver)
        if self.optimize_batch_size:
            cs.add_hyperparameter(batch_size)
        if self.optimize_learning_rate_init:
            cs.add_hyperparameter(learning_rate_init)
        if self.optimize_max_iter:
            cs.add_hyperparameter(max_iter)

        if self.optimize_solver and self.optimize_learning_rate_init:
            use_lr_init = InCondition(
                cs=cs, child=learning_rate_init, parent=solver, values=["sgd", "adam"]
            )
            cs.add_condition(use_lr_init)
        if self.optimize_solver and self.optimize_batch_size:
            use_batch_size = InCondition(
                cs=cs, child=batch_size, parent=solver, values=["sgd", "adam"]
            )
            cs.add_condition(use_batch_size)
        return cs

    def train(self, config: Configuration, seed: int) -> float:
        """Train an MLP based on a configuration and evaluate it on the
        digit-dataset using cross-validation."""
        n_layer = config["n_layer"] if "n_layer" in config else 1
        n_neurons = config["n_neurons"] if "n_neurons" in config else 10
        activation = config["activation"] if "activation" in config else "tanh"
        solver = config["solver"] if "solver" in config else "adam"
        batch_size = config["batch_size"] if "batch_size" in config else 200
        lr_init = (
            config["learning_rate_init"] if "learning_rate_init" in config else 0.001
        )
        max_iter = config["max_iter"] if "max_iter" in config else 25

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = MLPClassifier(
                hidden_layer_sizes=n_neurons * n_layer,
                solver=solver,
                batch_size=batch_size,
                activation=activation,
                learning_rate="constant",
                learning_rate_init=lr_init,
                max_iter=max_iter,
                random_state=self.seed,
            )

            cv = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
            score = cross_val_score(
                classifier, digits.data, digits.target, cv=cv, error_score="raise"
            )

        return 1 - np.mean(score)


class BDT:
    def __init__(
        self,
        optimize_learning_rate=False,
        optimize_n_estimators=False,
        seed=0,
    ):
        self.optimize_learning_rate = optimize_learning_rate
        self.optimize_n_estimators = optimize_n_estimators
        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)

        learning_rate = Float(
            "learning_rate", (0.0001, 1.0), default=0.1, log=True
        )
        n_estimators = Integer("n_estimators", (1, 200), default=100)

        if self.optimize_learning_rate:
            cs.add_hyperparameter(learning_rate)
        if self.optimize_n_estimators:
            cs.add_hyperparameter(n_estimators)

        return cs

    def train(self, config: Configuration, seed: int) -> float:
        """Train an Ada Boost Classifier based on a configuration and evaluate it on the
        digit-dataset using cross-validation."""

        learning_rate = (
            config["learning_rate"] if "learning_rate" in config else 0.1
        )
        n_estimators = (
            config["n_estimators"] if "n_estimators" in config else 100
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = AdaBoostClassifier(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=self.seed,
            )

            cv = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
            score = cross_val_score(
                classifier, digits.data, digits.target, cv=cv, error_score="raise"
            )

        return 1 - np.mean(score)


class DT:
    def __init__(
        self,
        optimize_max_depth=False,
        optimize_min_samples_leaf=False,
        seed=0,
    ):
        self.optimize_max_depth = optimize_max_depth
        self.optimize_min_samples_leaf = optimize_min_samples_leaf
        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)

        max_depth = Integer(
            "max_depth", (1, 20), default=3, 
        )
        min_samples_leaf = Integer("min_samples_leaf", (1, 100), default=1)

        if self.optimize_max_depth:
            cs.add_hyperparameter(max_depth)
        if self.optimize_min_samples_leaf:
            cs.add_hyperparameter(min_samples_leaf)

        return cs

    def train(self, config: Configuration, seed: int) -> float:
        """Train a Decision Tree Classifier based on a configuration and evaluate it on the
        digit-dataset using cross-validation."""

        max_depth = (
            config["max_depth"] if "max_depth" in config else 3
        )
        min_samples_leaf = (
            config["min_samples_leaf"] if "min_samples_leaf" in config else 1
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=self.seed,
            )

            cv = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
            score = cross_val_score(
                classifier, digits.data, digits.target, cv=cv, error_score="raise"
            )

        return 1 - np.mean(score)


class SVM:
    def __init__(
        self,
        optimize_kernel=False,
        optimize_C=False,
        optimize_shrinking=False,
        optimize_degree=False,
        optimize_coef=False,
        optimize_gamma=False,
        seed=0,
    ):
        self.optimize_kernel = optimize_kernel
        self.optimize_C = optimize_C
        self.optimize_shrinking = optimize_shrinking
        self.optimize_degree = optimize_degree
        self.optimize_coef = optimize_coef
        self.optimize_gamma = optimize_gamma
        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        kernel = Categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"], default="poly"
        )
        C = Float("C", (0.15, 1000.0), default=1.0, log=True)
        shrinking = Categorical("shrinking", [True, False], default=True)
        degree = Integer("degree", (1, 5), default=3)
        coef = Float("coef0", (0.0, 10.0), default=0.0)
        gamma = Float("gamma", (0.0001, 8.0), default=1.0, log=True)

        if self.optimize_kernel:
            cs.add_hyperparameter(kernel)
        if self.optimize_C:
            cs.add_hyperparameter(C)
        if self.optimize_shrinking:
            cs.add_hyperparameter(shrinking)
        if self.optimize_degree:
            cs.add_hyperparameter(degree)
        if self.optimize_coef:
            cs.add_hyperparameter(coef)
        if self.optimize_gamma:
            cs.add_hyperparameter(gamma)

        if self.optimize_kernel and self.optimize_degree:
            use_degree = InCondition(
                cs=cs, child=degree, parent=kernel, values=["poly"]
            )
            cs.add_condition(use_degree)
        if self.optimize_kernel and self.optimize_coef:
            use_coef = InCondition(
                cs=cs, child=coef, parent=kernel, values=["poly", "sigmoid"]
            )
            cs.add_condition(use_coef)
        if self.optimize_kernel and self.optimize_gamma:
            use_gamma = InCondition(
                cs=cs, child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]
            )
            cs.add_condition(use_gamma)
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Train an SVM based on a configuration and evaluate it on the
        digits-dataset using cross-validation."""
        C = config["C"] if "C" in config else 1.0
        shrinking = config["shrinking"] if "shrinking" in config else True
        degree = config["degree"] if "degree" in config else 3
        coef = config["coef"] if "coef" in config else 0.0
        gamma = config["gamma"] if config["gamma"] else 1.0

        classifier = svm.SVC(
            C=C,
            shrinking=shrinking,
            degree=degree,
            coef0=coef,
            gamma=gamma,
            random_state=self.seed,
        )
        scores = cross_val_score(classifier, digits.data, digits.target, cv=5) # score: accuracy
        cost = 1 - np.mean(scores)

        return cost
