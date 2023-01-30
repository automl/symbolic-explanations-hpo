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


iris = datasets.load_iris()
dataset = load_digits()


class SVM:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        kernel = Categorical("kernel", ["linear", "poly", "rbf", "sigmoid"], default="poly")
        C = Float("C", (0.15, 1000.0), default=1.0, log=True)
        shrinking = Categorical("shrinking", [True, False], default=True)
        degree = Integer("degree", (1, 5), default=3)
        coef = Float("coef0", (0.0, 10.0), default=0.0)
        gamma = Categorical("gamma", ["auto", "value"], default="value")
        gamma_value = Float("gamma_value", (0.0001, 8.0), default=1.0, log=True)

        # Then we create dependencies
        use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
        use_coef = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
        use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
        use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])

        # Add hyperparameters and conditions to our configspace
        cs.add_hyperparameters([C])
        #cs.add_hyperparameters([kernel, C, shrinking, degree, coef, gamma, gamma_value])
        #cs.add_conditions([use_degree, use_coef, use_gamma, use_gamma_value])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Creates an SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = config.get_dictionary()
        if "gamma" in config:
            config_dict["gamma"] = config_dict["gamma_value"] if config_dict["gamma"] == "value" else "auto"
            config_dict.pop("gamma_value", None)

        classifier = svm.SVC(**config_dict, random_state=seed)
        scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
        cost = 1 - np.mean(scores)

        return cost


class MLP():
    def __init__(self, optimize_n_neurons=False, optimize_n_layer=False, optimize_activation=False,
                 optimize_solver=False, optimize_batch_size=False, optimize_learning_rate=False,
                 optimize_learning_rate_init=False, max_iter=25, seed=0):
        self.optimize_n_neurons = optimize_n_neurons
        self.optimize_n_layer = optimize_n_layer
        self.optimize_activation = optimize_activation
        self.optimize_solver = optimize_solver
        self.optimize_batch_size = optimize_batch_size
        self.optimize_learning_rate = optimize_learning_rate
        self.optimize_learning_rate_init = optimize_learning_rate_init
        self.max_iter = max_iter
        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()
        cs.seed(self.seed)

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical("activation", ["logistic", "tanh", "relu"], default="tanh")
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

        if self.optimize_n_layer:
            cs.add_hyperparameter(n_layer)
        if self.optimize_n_neurons:
            cs.add_hyperparameter(n_neurons)
        if self.optimize_activation:
            cs.add_hyperparameter(activation)
        if self.optimize_solver:
            cs.add_hyperparameter(solver)
        if self.optimize_learning_rate:
            cs.add_hyperparameter(learning_rate)
        if self.optimize_batch_size:
            cs.add_hyperparameter(batch_size)
        if self.optimize_learning_rate_init:
            cs.add_hyperparameter(learning_rate_init)

        # Adding conditions to restrict the hyperparameter space...
        # ... since learning rate is only used when solver is 'sgd'.
        if self.optimize_solver and self.optimize_learning_rate:
            use_lr = EqualsCondition(cs=cs, child=learning_rate, parent=solver, value="sgd")
            cs.add_condition(use_lr)
        # ... since learning rate initialization will only be accounted for when using 'sgd' or 'adam'.
        if self.optimize_solver and self.optimize_learning_rate_init:
            use_lr_init = InCondition(cs=cs, child=learning_rate_init, parent=solver, values=["sgd", "adam"])
            cs.add_condition(use_lr_init)
        # ... since batch size will not be considered when optimizer is 'lbfgs'.
        if self.optimize_solver and self.optimize_batch_size:
            use_batch_size = InCondition(cs=cs, child=batch_size, parent=solver, values=["sgd", "adam"])
            cs.add_condition(use_batch_size)
        return cs

    def train(self, config: Configuration, seed: int) -> float:
        # For deactivated parameters (by virtue of the conditions),
        # the configuration stores None-values.
        # This is not accepted by the MLP, so we replace them with placeholder values.
        n_layer = config["n_layer"] if "n_layer" in config else 1
        n_neurons = config["n_neurons"] if "n_neurons" in config else 10
        activation = config["activation"] if "activation" in config else "tanh"
        solver = config["solver"] if "solver" in config else "adam"
        batch_size = config["batch_size"] if "batch_size" in config else 200
        lr = config["learning_rate"] if "learning_rate" in config else "constant"
        lr_init = config["learning_rate_init"] if "learning_rate_init" in config else 0.001

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = MLPClassifier(
                hidden_layer_sizes=n_neurons * n_layer,
                solver=solver,
                batch_size=batch_size,
                activation=activation,
                learning_rate=lr,
                learning_rate_init=lr_init,
                max_iter=self.max_iter,
                random_state=self.seed,
            )

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
            score = cross_val_score(classifier, dataset.data, dataset.target, cv=cv, error_score="raise")

        return 1 - np.mean(score)
