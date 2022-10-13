import numpy as np
from ..utils import to_string


class LatentModel:
    """
    Latent Model class can be used to quickly switch between different latent models.

    Parameters
    ----------
    type : str
        Type of latent model. Possible values are:
        - "sin_cos"
    model_kwargs : dict
        Keyword arguments for the latent model. See
        :py:func:`_latent_sin_cos_model` for details.
    """

    def __init__(self, type: str, **model_kwargs):

        self.type = type
        # Define all possible latent models
        switcher = {
            "sin_cos": _latent_sin_cos_model,
        }
        if type in switcher:
            self.model = switcher[type]
            self.model_kwargs = model_kwargs
        else:
            raise ValueError(f"Unknown type: {type}")

    def __call__(self, t):
        """
        Make model callable
        to be used e.g. in firing rates function.
        """
        return self.model(t, **self.model_kwargs)

    def __repr__(self):
        """Nice print in console"""
        return to_string(self.__dict__)


def _latent_sin_cos_model(t, omega=0.2):
    """
    Generate latent state from sinusoidal and cosine signals

    .. math::
        z_1(t) = \sin(\pi \omega t)
        z_2(t) = \cos(\pi \omega t)

    Parameters
    ----------
    t : 1D array, shape (n_measurements,)
        time
    omega : float
        frequency

    Returns
    -------
    latent_state : 2D array, shape (n_components, n_measurements)
        First dimension are the components/neurons, second dimension are the measurements/time.
    """

    latent_state = np.array([np.sin(np.pi * omega * t), np.cos(np.pi * omega * t)])
    return latent_state
