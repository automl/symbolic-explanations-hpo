import numpy as np
from typing import Callable
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter


class NamedFunction:
    def __init__(self, name: str, expression: str, function: Callable,
                 params: dict[str, tuple[float, float]]):
        """
        This class serves as a base class for functions. Each callable function is associated with a name
        and a (string) function expression.

        Parameters
        ----------
        name: Name of the function.
        expression: Function expression.
        function: Callable function.
        params: Dictionary containing hyperparameter names with their according min and max values.
        """
        self.name = name
        self.expression = expression
        self.function = function
        cs = ConfigurationSpace(seed=0)
        configs = []
        for name, (x_min, x_max) in params.items():
            configs.append(UniformFloatHyperparameter(name, lower=x_min, upper=x_max))
        cs.add_hyperparameters(configs)
        self.cs = cs

    def apply(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Apply the callable function to a float or array.
        """
        return self.function(x)

    def smac_apply(self, config: ConfigurationSpace, seed: int) -> float:
        """
        Function to be passed as target function for SMAC (requires config and seed).
        """
        hps = [config[hp] for hp in config.keys()]
        if len(hps) > 1:
            # input a list to the function in the multivariate case
            return self.function(hps)
        else:
            # input a float into to function in the univariate case
            return self.function(hps[0])


def get_functions1d() -> list[NamedFunction]:
    """
    Creates a list containing instantiated 1D NamedFunction objects.
    To evaluate on more 1D functions, add additional functions here.
    Warning: Use unique names to avoid overriding.
    """
    params = {"x": (0.0, 1.0)}
    functions = [
        NamedFunction("Quadratic function A", "x**2", lambda x: x ** 2, params),
        NamedFunction("Quadratic function B", "(x-1/2)**2", lambda x: (x - 0.5) ** 2, params),
        NamedFunction("Polynom function A", "x**3", lambda x: x ** 3, params),
        NamedFunction("Polynom function B", "x**4", lambda x: x ** 4, params),
        NamedFunction("Square root function", "x**(1/2)", lambda x: x ** (1 / 2), params),
        NamedFunction("Inv square root function", "x**(-1/2)", lambda x: x ** (-1 / 2), params),
        NamedFunction("Sinosoidal function A", "sin(x)", lambda x: np.sin(x), params),
        NamedFunction("Sinosoidal function B", "sin(2*x)", lambda x: np.sin(2 * x), params),
        NamedFunction("Exponential function A", "exp(x)", lambda x: np.exp(x), params),
        NamedFunction("Exponential function B", "3*exp(x)", lambda x: 3 * np.exp(x), params),
        NamedFunction("Exponential function C", "exp(-3*x)", lambda x: np.exp(-3 * x), params),
        NamedFunction(
            "Rational function", "x/(x + 1)**2", lambda x: x / ((x + 1) ** 2), params
        ),
    ]
    return functions


def get_functions2d() -> list[NamedFunction]:
    """
    Creates a list containing instantiated 2D NamedFunction objects.
    To evaluate on more 2D functions, add additional functions here.
    Warning: Use unique names to avoid overriding.
    """
    functions = [
        NamedFunction("Linear 2D", "X0 + 2*X1", lambda x: x[0] + 2 * x[1],
                      params=({"x0": (-10, 10), "x1": (-5, 5)})
                      ),
        NamedFunction(
            "Polynom function 2D", "2*X0 + X1**2", lambda x: 2 * x[0] + x[1]**2,
            params=({"X0": (-10, 10), "X1": (-5, 5)})
        ),
        NamedFunction(
            "Exponential function 2D",
            "2*exp(X0) + exp(3*X1)",
            lambda x: 2 * np.exp(x[0]) + np.exp(3 * x[1]),
            params=({"X0": (-10, 10), "X1": (-5, 5)})
        ),
        NamedFunction(
            "Rosenbrock 2D",
            "100 * (X1 - X0**2)**2.0 + (1 - X0)**2",
            lambda x: 100.0 * (x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0,
            params=({"X0": (-10, 10), "X1": (-5, 5)})
        ),
        NamedFunction(
            "Branin 2D",
            "X1 - 5.1 / (4 * PI**2) * X0**2 + 5 / PI * X0 - 6)**2 + 10 * (1 - 1 / (8 * PI)) * cos(X0) + 10",
            lambda x: (x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
                        1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10,
            params=({"X0": (-5, 10), "X1": (0, 15)})
        ),
        NamedFunction(
            "Camelback 2D",
            "(4 - 2.1 * X0**2 + X0**4 / 3) * X0**2 + X0 * X1 + (-4 + 4 * X1**2) * X1**2",
            lambda x: (4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2,
            params=({"X0": (-3, 3), "X1": (-2, 2)})
        ),

    ]
    return functions
