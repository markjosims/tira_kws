from numba import jit, prange
import numpy as np
from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series

@jit(nopython=True, cache=True)
def batched_subseq_dtw(
    batched_distances: np.ndarray,
    query_lengths: np.ndarray,
    reference_lengths: np.ndarray,
) -> np.ndarray:
    """
    Computes subsequence DTW distances for a batch of sequences.
    The input is a 3D array of shape (batch_size, max_query_length,
    max_reference_length) containing the pairwise distances between
    query and reference sequences. The output is an array of shape
    (batch_size) containing the minimum DTW distance for each sequence
    pair in the batch.

    Arguments:
        batched_distances: A 3D numpy array of shape (batch_size,
            max_query_length, max_reference_length) containing the pairwise
            distances between query and reference sequences.
        query_lengths: A 1D numpy array of shape (batch_size,) containing
            the actual lengths of the query sequences.
        reference_lengths: A 1D numpy array of shape (batch_size,) containing
            the actual lengths of the reference sequences.

    Returns:
        dtw_scores: A 1D numpy array of shape (batch_size,) containing the minimum DTW
        distance for each sequence pair in the batch.
    """
    if not (batched_distances.ndim == 3):
        raise ValueError(
            f"Expected batched_distances to be a 3D array, got {batched_distances.ndim}D array"
        )
    if not (query_lengths.ndim == 1 and reference_lengths.ndim == 1):
        raise ValueError(
            f"Expected query_lengths and reference_lengths to be 1D arrays, got {query_lengths.ndim}D and {reference_lengths.ndim}D arrays"
        )
    if not (batched_distances.shape[0] == query_lengths.shape[0] == reference_lengths.shape[0]):
        raise ValueError(
            f"Batch size mismatch: expected {batched_distances.shape[0]}, got {query_lengths.shape[0]} and {reference_lengths.shape[0]}"
        )

    batch_size = batched_distances.shape[0]
    dtw_scores = np.zeros(batch_size, dtype=np.float64)

    for b in prange(batch_size):
        query_len = query_lengths[b]
        seq_len = reference_lengths[b]
        dist_mat = batched_distances[b]

                # for subsequence DTW, set the first column to 0
        # (allowing for starting at any point in the reference sequence)
        cost = np.full(
            (query_len + 1, seq_len + 1),
            np.inf,
            dtype=np.float64,
        )
        cost[0, :] = 0

        # Fill the cost matrix
        for i in range(1, query_len + 1):
            for j in range(1, seq_len + 1):
                insert_cost = cost[i - 1, j]
                delete_cost = cost[i, j - 1]
                match_cost = cost[i - 1, j - 1]
                min_cost = min(insert_cost, delete_cost, match_cost)
                current_distance = dist_mat[i - 1, j - 1]
                cost[i, j] = current_distance + min_cost
        min_cost = np.min(cost[query_len,1:seq_len+1])  # minimum cost over the last row
        dtw_scores[b] = min_cost
    
    return dtw_scores

"""
The following code was copied from [tslearn/metrics/dtw_variants.py](https://github.com/tslearn-team/tslearn/blob/fae377a1c21a03cbe1f656364361b19a85233323/tslearn/metrics/dtw_variants.py#L1144)
on 19 Dec 2025 and modified to use cosine distance rather than Euclidean.
"""

def dtw_subsequence_path(subseq, longseq, be=None):
    r"""Compute sub-sequence Dynamic Time Warping (DTW) similarity measure
    between a (possibly multidimensional) query and a long time series and
    return both the path and the similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Compared to traditional DTW, here, border constraints on admissible paths
    :math:`\pi` are relaxed such that :math:`\pi_0 = (0, ?)` and
    :math:`\pi_L = (N-1, ?)` where :math:`L` is the length of the considered
    path and :math:`N` is the length of the subsequence time series.

    It is not required that both time series share the same size, but they must
    be the same dimension. This implementation finds the best matching starting
    and ending positions for `subseq` inside `longseq`.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d) or (sz1,)
        A query time series.
        If shape is (sz1,), the time series is assumed to be univariate.
    longseq : array-like, shape=(sz2, d) or (sz2,)
        A reference (supposed to be longer than `subseq`) time series.
        If shape is (sz2,), the time series is assumed to be univariate.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`.
    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_subsequence_path([2., 3.], [1., 2., 2., 3., 4.])
    >>> path
    [(0, 2), (1, 3)]
    >>> float(dist)
    0.0

    See Also
    --------
    dtw : Get the similarity score for DTW
    subsequence_cost_matrix: Calculate the required cost matrix
    subsequence_path: Calculate a matching path manually
    """
    be = instantiate_backend(be, subseq, longseq)
    subseq = to_time_series(subseq, be=be)
    longseq = to_time_series(longseq, be=be)
    acc_cost_mat = subsequence_cost_matrix(subseq=subseq, longseq=longseq, be=be)
    global_optimal_match = be.argmin(acc_cost_mat[-1, :])
    path = subsequence_path(acc_cost_mat, global_optimal_match, be=be)
    return path, acc_cost_mat[-1, :][global_optimal_match]

def subsequence_cost_matrix(subseq, longseq, be=None):
    """Compute the accumulated cost matrix score between a subsequence and
    a reference time series.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d) or (sz1,)
        Subsequence time series. If shape is (sz1,), the time series is assumed to be univariate.
    longseq : array-like, shape=(sz2, d) or (sz2,)
        Reference time series. If shape is (sz2,), the time series is assumed to be univariate.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    be = instantiate_backend(be, subseq, longseq)
    subseq = be.array(subseq)
    longseq = be.array(longseq)
    subseq = to_time_series(subseq, remove_nans=True, be=be)
    longseq = to_time_series(longseq, remove_nans=True, be=be)
    return _subsequence_cost_matrix(subseq, longseq, be=be)

def _subsequence_cost_matrix(subseq, longseq, be=None):
    """Compute the accumulated cost matrix score between a subsequence and
    a reference time series.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d)
        Subsequence time series.
    longseq : array-like, shape=(sz2, d)
        Reference time series.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    be = instantiate_backend(be, subseq, longseq)
    subseq = be.array(subseq)
    longseq = be.array(longseq)
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = be.full((l1 + 1, l2 + 1), be.inf)
    cum_sum[0, :] = 0.0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _local_dist(subseq[i], longseq[j], be=be)
            cum_sum[i + 1, j + 1] += min(
                cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]
            )
    return cum_sum[1:, 1:]

def _local_dist(x, y, be=None):
    """Compute the distance between two vectors.

    Parameters
    ----------
    x : array-like, shape=(d,)
        A vector.
    y : array-like, shape=(d,)
        Another vector.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    dist : float
        Cosine distance between x and y.
    """
    # BEGIN MODIFICATION
    # use cosine distance instead of Euclidean
    # avoid tslearn backend interface since all arrays
    # in this project use Torch tensors
    x /= np.linalg.norm(x, ord=2)
    y /= np.linalg.norm(y, ord=2)
    dist = 1-np.dot(x, y)
    # END MODIFICATION

    return dist

def subsequence_path(acc_cost_mat, idx_path_end, be=None):
    r"""Compute the optimal path through an accumulated cost matrix given the
    endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array-like, shape=(sz1, sz2)
        Accumulated cost matrix comparing subsequence from a longer sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`

    Examples
    --------

    >>> acc_cost_mat = numpy.array([[1., 0., 0., 1., 4.],
    ...                             [5., 1., 1., 0., 1.]])
    >>> # calculate the globally optimal path
    >>> optimal_end_point = numpy.argmin(acc_cost_mat[-1, :])
    >>> path = subsequence_path(acc_cost_mat, optimal_end_point)
    >>> path
    [(0, 2), (1, 3)]

    See Also
    --------
    dtw_subsequence_path : Get the similarity score for DTW
    subsequence_cost_matrix: Calculate the required cost matrix

    """
    be = instantiate_backend(be, acc_cost_mat)
    acc_cost_mat = be.array(acc_cost_mat)

    return _subsequence_path(acc_cost_mat, idx_path_end, be=be)

def _subsequence_path(acc_cost_mat, idx_path_end, be=None):
    r"""Compute the optimal path through an accumulated cost matrix given the
    endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array-like, shape=(sz1, sz2)
        Accumulated cost matrix comparing subsequence from a longer sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`
    """
    be = instantiate_backend(be, acc_cost_mat)
    acc_cost_mat = be.array(acc_cost_mat)
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, idx_path_end)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = be.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = be.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]