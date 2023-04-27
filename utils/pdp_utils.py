import numpy as np
from utils.run_utils import get_surrogate_predictions


def get_pdp(X_train, cs, surrogate_model, idx, n_ice):
    optimized_parameters = cs.get_hyperparameters()
    n_grid = len(X_train)
    ice_samples = np.array(
        [list(i.get_dictionary().values()) for i in cs.sample_configuration(size=n_ice)]
    )

    x_ice = ice_samples.repeat(n_grid)
    x_ice = x_ice.reshape((n_ice, len(optimized_parameters), n_grid))
    x_ice = x_ice.transpose((0, 2, 1))
    x_ice[:, :, idx] = X_train

    y_ice = np.array(get_surrogate_predictions(x_ice.reshape(-1, 4), cs, surrogate_model))
    y_ice = y_ice.reshape((n_ice, n_grid))
    pdp = y_ice.mean(axis=0)
    return pdp