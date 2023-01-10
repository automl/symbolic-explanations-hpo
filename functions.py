import numpy as np
from typing import Callable
from ConfigSpace import ConfigurationSpace


class NamedFunction:
    def __init__(self, name: str, expression: str, function: Callable):
        """
        This class serves as a base class for functions. Each callable function is associated with a name
        and a (string) function expression.

        Parameters
        ----------
        name: Name of the function.
        expression: Function expression.
        function: Callable function.
        """
        self.name = name
        self.expression = expression
        self.function = function

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
    functions = [
        NamedFunction("Quadratic function A", "x**2", lambda x: x**2),
        NamedFunction("Quadratic function B", "(x-1/2)**2", lambda x: (x - 0.5) ** 2),
        NamedFunction("Polynom function A", "x**3", lambda x: x**3),
        NamedFunction("Polynom function B", "x**4", lambda x: x**4),
        NamedFunction("Square root function", "x**(1/2)", lambda x: x ** (1 / 2)),
        NamedFunction("Inv square root function", "x**(-1/2)", lambda x: x ** (-1 / 2)),
        NamedFunction("Sinosoidal function A", "sin(x)", lambda x: np.sin(x)),
        NamedFunction("Sinosoidal function B", "sin(2*x)", lambda x: np.sin(2 * x)),
        NamedFunction("Exponential function A", "exp(x)", lambda x: np.exp(x)),
        NamedFunction("Exponential function B", "3*exp(x)", lambda x: 3 * np.exp(x)),
        NamedFunction("Exponential function C", "exp(-3*x)", lambda x: np.exp(-3 * x)),
        NamedFunction(
            "Rational function", "x/(x + 1)**2", lambda x: x / ((x + 1) ** 2)
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
        NamedFunction("Linear 2D", "X0 + 2*X1", lambda x: x[0] + 2 * x[1]),
        NamedFunction(
            "Polynom function 2D", "2*X0 + X1**2", lambda x: 2 * x[0] + x[1] ** 2
        ),
        NamedFunction(
            "Exponential function 2D",
            "2*exp(X0) + exp(3*X1)",
            lambda x: 2 * np.exp(x[0]) + np.exp(3 * x[1]),
        ),
        NamedFunction(
            "Rosenbrock 2D",
            "100 * (X1 - X0**2) ** 2.0 + (1 - X0)**2",
            lambda x: 100.0 * (x[1] - x[0] ** 2.0) ** 2.0 + (1 - x[0]) ** 2.0,
        ),
    ]
    return functions
