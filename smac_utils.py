import logging
import numpy as np
from pathlib import Path
from ConfigSpace import ConfigurationSpace
from typing import Type
from smac import Scenario
from smac.facade import AbstractFacade


def run_smac_optimization(
    configspace: ConfigurationSpace,
    facade: Type[AbstractFacade],
    target_function,
    function_name: str,
    n_eval: int,
    run_dir: str,
    seed: int
) -> [np.ndarray, np.ndarray]:
    """Runs SMAC Hyperparameter Optimization on the given function within the hyperparameter space.

    Parameters
    ----------
    configspace : Hyperparameter configuration space.
    facade: SMAC facade to be used.
    target_function : Function to be minimized.
    function_name: Name of the function to be minimized.
    n_eval : Desired number of function evaluations.
    run_dir : Run directory to save SMAC output to.
    seed : Seed to be used in SMAC scenario.

    Returns
    ----------
    conf_hp: Evaluated hyperparameter settings.
    conf_res: According true function value for each evaluated hyperparameter setting.
    """

    scenario = Scenario(
        configspace=configspace,
        deterministic=True,
        n_trials=n_eval,
        output_directory=Path(f"{run_dir}/smac/{function_name}"),
        seed=seed
    )

    config_selector = facade.get_config_selector(scenario, retrain_after=1)

    smac = facade(
        scenario=scenario,
        target_function=target_function,
        logging_level=Path("logging_smac.yml"),
        config_selector=config_selector,
    )

    # re-add log dir handler to logger as it is destroyed everytime a new SMAC facade is created
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(filename=f"{run_dir}/log.log", encoding="utf8")
    handler.setLevel("INFO")
    handler.setFormatter(
        logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d] %(message)s")
    )
    logger.root.addHandler(handler)

    incumbent = smac.optimize()

    # get hyperparameter settings and corresponding function values from smac runhistory
    conf_hp, conf_res = [], []
    hp_names = configspace.get_hyperparameter_names()
    for hp_name in hp_names:
        conf_hp.append(
            [
                config.get_dictionary()[hp_name]
                if hp_name in config.get_dictionary() else None
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

    return conf_hp, conf_res.reshape(-1)
