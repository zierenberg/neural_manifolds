import sklearn.decomposition

def PCA(signal):
    """
    Makes a PCA
    Parameters
    ----------
    signal : 2D array, shape (n_components, n_measurements)
        First dimension are the components/neurons, second dimension are the
        measurements/time.

    Returns
    -------
    explained_variance_ratio : 1D array, shape (n_components,)
    PCA_object : : sklearn.decomposition.PCA


    """
    pca = sklearn.decomposition.PCA()
    pca.fit(signal)
    return pca.explained_variance_ratio_, pca
