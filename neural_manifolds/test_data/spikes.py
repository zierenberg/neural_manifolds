from bisect import bisect
import numpy as np
from ..utils import set_seed


@set_seed()
def inhomogenous_poisson_spikes(rate_object):
    """
    generate spikes from `n` (number of rows in rate_object["rates"]) time-continuous point processes with time-dependent rates `rate_i` (ith row of rate_object["rates"])

    Parameters
    ----------
    rate_object: dictionary with elements "rates" and "times". "rates" is a 2D array, shape (n_components, n_measurements) where 
        first dimension is neurons and second dimension is the time-dependent rate. "times" is a sorted 1D array of discrete times.

    Returns
    -------
    signals : 2D array, shape (id_neuron, time_spike)
    """
    # start with empty array
    #rates = np.array(rates)
    times = rate_object["times"]
    rates = rate_object["rates"]

    # Iterate over the "neurons"
    spikes = []
    for i, rate in enumerate(rates):
        rate_max = rate.max()
        # advance time continously as if it was a Poisson process with rate_max
        time = times[0] + np.random.exponential() / rate_max
        while time < times[-1]:
            # accept new time as a spike with probability rate(time)/rate_max
            time_index = bisect(times,time)-1
            if np.random.random() < rate[time_index] / rate_max:
                spikes.append([i, time])

            time += np.random.exponential() / rate_max
            
    return np.array(spikes)
