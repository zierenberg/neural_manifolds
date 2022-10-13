from ..utils import set_seed
import numpy as np

from .latent_model import LatentModel


@set_seed()
def firing_rates(
    latent_model: LatentModel,
    t: np.ndarray,
    N: int,
    coefficients: np.ndarray = None,
    offset: np.ndarray = None,
) -> np.ndarray:
    """
    Generate firing rates from latent model using a linear combination.

    .. math::
        \nu_i(t) = \sum_{j}^2 a_{j} z_j(t) + \mathrm{offset}

    Parameters
    ----------
    latent_model : callable
        function that generates latent state from time
    t : 1D array, shape (n_measurements,)
        time
    N : int
        Number of neurons
    coefficients : 2D array, shape (n_components, n_neurons)
        Coefficients for linear combination of latent state. If None, random
        coefficients are generated.

    Returns
    -------
    rates : 2D array, shape (n_components, n_measurements)
        First dimension are the components/neurons, second dimension are the measurements/time.
    """

    # Set default values
    if coefficients is None:
        coefficients = 2 * np.random.randn(N, 2)
    if offset is None:
        mean_rate = 20
        offset = mean_rate + 2 * np.random.randn(N)

    # Generate latent state
    latent_state = latent_model(t)

    # Generate firing rates by linear combination
    rates = np.dot(coefficients, latent_state) + offset[:, None]

    return rates
