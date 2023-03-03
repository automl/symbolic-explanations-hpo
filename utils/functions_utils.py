import numpy as np
from typing import Callable
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter


class NamedFunction:
    def __init__(
        self,
        name: str,
        expression: str,
        function: Callable,
        params: dict[str, tuple[float, float]],
    ):
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
        configspace = ConfigurationSpace(seed=0)
        configs = []
        for name, (x_min, x_max) in params.items():
            configs.append(UniformFloatHyperparameter(name, lower=x_min, upper=x_max))
        configspace.add_hyperparameters(configs)
        self.configspace = configspace

    def apply(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Apply the callable function to a float or array.
        """
        return self.function(x)

    def train(self, config: ConfigurationSpace, seed: int) -> float:
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
    params = {"x": (0.01, 1.0)}
    functions = [
        NamedFunction(
            name="Quadratic function A",
            expression="x**2",
            function=lambda x: x**2,
            params=params,
        ),
        NamedFunction(
            name="Quadratic function B",
            expression="(x-1/2)**2",
            function=lambda x: (x - 0.5) ** 2,
            params=params,
        ),
        NamedFunction(
            name="Polynom function A",
            expression="x**3",
            function=lambda x: x**3,
            params=params,
        ),
        NamedFunction(
            name="Polynom function B",
            expression="x**4",
            function=lambda x: x**4,
            params=params,
        ),
        NamedFunction(
            name="Square root function",
            expression="x**(1/2)",
            function=lambda x: x ** (1 / 2),
            params=params,
        ),
        NamedFunction(
            name="Inv square root function",
            expression="x**(-1/2)",
            function=lambda x: x ** (-1 / 2),
            params=params,
        ),
        NamedFunction(
            "Sinosoidal function A",
            expression="sin(x)",
            function=lambda x: np.sin(x),
            params=params,
        ),
        NamedFunction(
            name="Sinosoidal function B",
            expression="sin(2*x)",
            function=lambda x: np.sin(2 * x),
            params=params,
        ),
        NamedFunction(
            "Exponential function A",
            expression="exp(x)",
            function=lambda x: np.exp(x),
            params=params,
        ),
        NamedFunction(
            name="Exponential function B",
            expression="3*exp(x)",
            function=lambda x: 3 * np.exp(x),
            params=params,
        ),
        NamedFunction(
            name="Exponential function C",
            expression="exp(-3*x)",
            function=lambda x: np.exp(-3 * x),
            params=params,
        ),
        NamedFunction(
            name="Rational function",
            expression="x/(x + 1)**2",
            function=lambda x: x / ((x + 1) ** 2),
            params=params,
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
        NamedFunction(
            name="Linear 2D",
            expression="X0 + 2*X1",
            function=lambda x: x[0] + 2 * x[1],
            params=({"X0": (-10, 10), "X1": (-5, 5)}),
        ),
        NamedFunction(
            name="Polynom function 2D",
            expression="2*X0 + X1**2",
            function=lambda x: 2 * x[0] + x[1] ** 2,
            params=({"X0": (-10, 10), "X1": (-5, 5)}),
        ),
        NamedFunction(
            name="Exponential function 2D",
            expression="2*exp(X0) + exp(3*X1)",
            function=lambda x: 2 * np.exp(x[0]) + np.exp(3 * x[1]),
            params=({"X0": (-2, 2), "X1": (-2, 2)}),
        ),
        NamedFunction(
            name="Rosenbrock 2D",
            expression="100 * (X1 - X0**2)**2.0 + (1 - X0)**2",
            function=lambda x: 100.0 * (x[1] - x[0] ** 2.0) ** 2.0 + (1 - x[0]) ** 2.0,
            params=({"X0": (-10, 10), "X1": (-5, 5)}),
        ),
        NamedFunction(
            name="Branin 2D",
            expression="(X1 - 5.1 / (4 * PI**2) * X0**2 + 5 / PI * X0 - 6)**2 + 10 * (1 - 1 / (8 * PI)) * cos(X0) + 10",
            function=lambda x: (
                x[1] - 5.1 / (4 * np.pi**2) * x[0] ** 2 + 5 / np.pi * x[0] - 6
            )
            ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0])
            + 10,
            params=({"X0": (-5, 10), "X1": (0, 15)}),
        ),
        NamedFunction(
            name="Camelback 2D",
            expression="(4 - 2.1 * X0**2 + X0**4 / 3) * X0**2 + X0 * X1 + (-4 + 4 * X1**2) * X1**2",
            function=lambda x: (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2
            + x[0] * x[1]
            + (-4 + 4 * x[1] ** 2) * x[1] ** 2,
            params=({"X0": (-3, 3), "X1": (-2, 2)}),
        ),
    ]
    return functions
