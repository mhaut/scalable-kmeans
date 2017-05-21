def _k_para_init(X, n_clusters, x_squared_norms, random_state, l=4, r=5):
	
    """Init n_clusters seeds according to k-means||
    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters: integer
        The number of seeds to choose
    
    l: int
        Number of centers to sample at each iteration of k-means||.
    
    r: int
        Number of iterations of k-means|| to perfoem.
    x_squared_norms: array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state: numpy.RandomState
        The generator used to initialize the centers.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Bahmani, Bahman, et al. "Scalable k-means++."
    Proceedings of the VLDB Endowment 5.7 (2012): 622-633.
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, _= X.shape

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        center = X[center_id].toarray()
    else:
        center = X[center_id, np.newaxis]

    # Initialize list of closest distances and calculate current potential
    dist_sq = euclidean_distances(center, X, \
				  Y_norm_squared=x_squared_norms, squared=True)
    cost = dist_sq.sum()

    center_ids = [center_id]

    r = max(n_clusters / l, r) 

    if l * r < n_clusters:
        raise ValueError('l * r should be greater or equal to n_clusters (l={}'
                         ', r={}, n_clusters={}).'.format(l, r, n_clusters))

    for _ in range(r):
        # Choose new centers by sampling with probability proportional
        # to the squared distance to the closest existing center
        # Approx. l*d(x, C) / cost
        rand_vals = random_state.random_sample(l) * cost
        candidate_ids = np.searchsorted(dist_sq.cumsum(), rand_vals)

        # Compute distances to new centers
	candidate_ids < n_samples
	distance_to_candidates = euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Compute potential when including the new centers
        distances = np.min(distance_to_candidates, axis=0)
        closest_dist_sq  = np.minimum(distances, closest_dist_sq)
        cost = closest_dist_sq.sum()

        center_ids.extend(candidate_ids)

    centers, _, _ = k_means(X[center_ids], n_clusters, init='k-means++')

    return centers
