from scipy.ndimage import gaussian_filter1d
import numpy as np

def smooth_spikes(spikes, bin_width = None, std_gaussian = 0.1, sampling_width=0.05, sqrt=True):
    """

    Parameters
    ----------
    spikes : 2D array, shape (id_neuron, time_spike)
    bin_width : float (in seconds)
        if doing a histogram, set a bin width. If set to None, it still does a
        histogram, but with very short bins of 4 ms
    std_gaussian : float (in seconds)
        Standard deviation of the smoothing kernel, can be set to None if no smoothing
        is wanted
    sampling_width : float (in seconds)
        The step width of the final sampling of the signal
    sqrt : boolean
        Whether to take the square root of the spike count

    Returns
    -------
    time : 1D array of the time
    signals : 2D array, shape (num_neurons, len_time)

    """


    num_neurons = len(np.unique(spikes[:,0])) #unique neuron ids
    max_time = np.max(spikes[:,1])

    assert not (bin_width is None and std_gaussian is None)

    if bin_width is None:
        bin_width = 0.004 # For simplicity, make simply 4 ms bins

    if bin_width is not None:
        time = np.arange(0,max_time+bin_width,bin_width)
        counts, _, bins = np.histogram2d(
            spikes[:,0], # ids
            spikes[:,1], # times
            bins=[
                np.arange(num_neurons+1), # neurons
                time, #times
                ]
        )
        time = time[:-1] # Because the last element is the end of the bin

        if sqrt:
            counts = np.sqrt(counts)
        if std_gaussian is not None:
            sd=std_gaussian/bin_width
            signal = gaussian_filter1d(counts, sigma=sd, axis=1)
        else:
            signal = counts

    if not np.abs((sampling_width / bin_width) % 1) < 0.01:
        raise RuntimeError(f"The sampling_width of {sampling_width} is not a "
                           f"multiple of the bin width of {bin_width}")

    subsampling_factor = int(np.round(sampling_width / bin_width))
    signal = signal[:, ::subsampling_factor]
    time = time[::subsampling_factor]

    return time, signal
