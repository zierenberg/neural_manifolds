import sklearn.decomposition
import sklearn.manifold


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


def LLE(signal, n_neighbors=12, n_components=2):
    """
    Locally_linear_embedding

    Parameters
    ----------

    Returns
    -------
    """
    lle = sklearn.manifold.LocallyLinearEmbedding(
        n_components=n_components, n_neighbors=n_neighbors
    )
    lle.fit(signal)
    return lle.embedding_, lle
