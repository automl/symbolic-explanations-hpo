import logging
import numpy as np
from pathlib import Path
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac import BlackBoxFacade, Scenario

from functions import NamedFunction


def run_smac_optimization(
    hp_space: dict[str, tuple[float, float]],
    function: NamedFunction,
    n_eval: int,
    run_dir: str,
) -> [np.ndarray, np.ndarray]:
    """Runs SMAC Hyperparameter Optimization on the given function within the hyperparameter space.

    Parameters
    ----------
    hp_space : Scenario.
    function : Function to be minimized.
    n_eval : Desired number of function evaluations.
    run_dir : Run directory to save SMAC output to.

    Returns
    ----------
    conf_hp: Evaluated hyperparameter settings.
    conf_res: According true function value for each evaluated hyperparameter setting.
    """

    configspace = ConfigurationSpace()
    for hp_name in hp_space.keys():
        hp_min = hp_space[hp_name][0]
        hp_max = hp_space[hp_name][1]
        configspace.add_hyperparameter(
            UniformFloatHyperparameter(hp_name, hp_min, hp_max)
        )

    scenario = Scenario(
        configspace=configspace,
        deterministic=True,
        n_trials=n_eval,
        output_directory=Path(
            f"{run_dir}/smac/{function.name.lower().replace(' ', '_')}"
        ),
    )

    smac = BlackBoxFacade(
        scenario=scenario,
        target_function=function.smac_apply,
        logging_level=Path("logging_smac.yml"),
    )

    # re-add log dir handler to logger as it is destroyed everytime a new SMAC facade is created
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(filename=f"{run_dir}/log.log", encoding="utf8")
    handler.setLevel("INFO")
    handler.setFormatter(
        logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    logger.root.addHandler(handler)
    logger.info(f"Run SMAC for: {function.name}: {function.expression}")

    incumbent = smac.optimize()

    # get hyperparameter settings and corresponding function values from smac runhistory
    conf_hp, conf_res = [], []
    for hp_name in hp_space.keys():
        conf_hp.append(
            [
                config.get_dictionary()[hp_name]
                for config in smac.runhistory.get_configs()
            ]
        )
        conf_res.append(
            [
                smac.runhistory.get_cost(config)
                for config in smac.runhistory.get_configs()
            ]
        )

    conf_hp, conf_res = np.array(conf_hp), np.array(conf_res)

    return conf_hp, conf_res
