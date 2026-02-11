from tslearn.metrics import dtw_path_from_metric
import torch
from src.encoding import get_cosine_distance
from typing import List
from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series

"""
Functions for performing Dynamic Time Warping on speech embeddings
"""

def pairwise_dtw(
        query_embeds: List[torch.Tensor],
        test_embeds: List[torch.Tensor]
) -> torch.Tensor:
    """
    Computes pairwise DTW similarity between a set of query and
    test phrase embeddings. Uses cosine similarity as a distance
    metric.

    Args:
        query_embeds: list of embedded query phrases
        test_embeds: list of embedded test phrases

    Returns:
        torch.Tensor of shape (Q*T) with the DTW similarity score of each
        query and test phrase
    """
    n_query = len(query_embeds)
    n_test = len(test_embeds)
    dtw_matrix = torch.zeros(n_query, n_test)
    for i in range(n_query):
        for j in range(n_test):
             _, dtw_score = dtw_path_from_metric(
                query_embeds[i],
                test_embeds[j],
                metric=get_cosine_distance,
             )
             # _, dtw_score = dtw_subsequence_path(
             #     query_embeds[i],
             #     test_embeds[j],
             # )
             # convert distance to similarity
             dtw_score = 1 - dtw_score
             dtw_matrix[i, j] = dtw_score

    return dtw_matrix

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
    return path, be.sqrt(acc_cost_mat[-1, :][global_optimal_match])

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
            cum_sum[i + 1, j + 1] = _local_squared_dist(subseq[i], longseq[j], be=be)
            cum_sum[i + 1, j + 1] += min(
                cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]
            )
    return cum_sum[1:, 1:]

def _local_squared_dist(x, y, be=None):
    """Compute the squared distance between two vectors.

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
        Squared distance between x and y.
    """
    # BEGIN MODIFICATION
    # use cosine distance instead of Euclidean
    # avoid tslearn backend interface since all arrays
    # in this project use Torch tensors
    x /= x.norm()
    y /= y.norm()
    dist = 1-torch.dot(x, y)
    dist = dist**2
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