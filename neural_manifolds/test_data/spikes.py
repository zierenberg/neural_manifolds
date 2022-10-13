import numpy as np
from ..utils import set_seed


@set_seed()
def inhomogenous_poisson_spikes(rates, dt):
    """
    generate spikes from `n` (number of rows in rates) time-continuous point processes with time-dependent rates `rate_i` (ith row of rates)

    Parameters
    ----------
    rates :  2D array, shape (n_components, n_measurements)
        First dimension are the components/neurons, second dimension are the measurements/time.
    dt : float
        time step

    Returns
    -------
    signals : 2D array, shape (id_neuron, time_spike)
    """
    # start with empty array
    rates = np.array(rates)
    t_max = rates.shape[1] * dt

    # Iterate over the "neurons"
    spikes = []
    for i, rate in enumerate(rates):
        rate_max = rate.max()
        # Can someone explain in more detail what this does?
        time = 0
        while time < t_max:
            time += np.random.exponential() / rate_max
            if time < t_max:
                if np.random.random() < rate[int(time / dt)] / rate_max:
                    spikes.append([i, time])

    return np.array(spikes)
