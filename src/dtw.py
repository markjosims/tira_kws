from numba import jit
import numpy as np

@jit
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
    # Initialize the cost matrix
    batch_size = batched_distances.shape[0]
    max_query_length = batched_distances.shape[1]
    max_reference_length = batched_distances.shape[2]
    cost = np.full(
        (batch_size, max_query_length + 1, max_reference_length + 1),
        np.inf,
        dtype=np.float64,
    )

    # for subsequence DTW, set the first column to 0
    # (allowing for starting at any point in the reference sequence)
    cost[:, 0, :] = 0

    for dist_mat, query_len, seq_len in zip(batched_distances, query_lengths, reference_lengths):

        # Fill the cost matrix
        for i in range(1, query_len + 1):
            for j in range(1, seq_len + 1):
                prev_costs = np.array([
                    cost[i - 1, j],    # insertion
                    cost[i, j - 1],    # deletion
                    cost[i - 1, j - 1] # match
                ], dtype=np.float64)
                cost[i, j] = dist_mat[i - 1, j - 1] + np.min(prev_costs)

    min_cost = np.min(cost[:, query_lengths, :], axis=1)
    return min_cost