from src.dtw import batched_subseq_dtw
from src.distance import pairwise_cosine_similarity, pad_arrays_or_tensors
from tests.test_utils import TEST_EMBED_DIM
from tslearn.metrics import dtw_subsequence_path
import pytest
from random import randint
import numpy as np
import torch

@pytest.mark.parametrize(
    "num_queries,num_tests", [(randint(5, 20), randint(5,20)) for _ in range(60)]
)
def test_batched_subseq_dtw(num_queries, num_tests):
    max_query_length = 10
    max_reference_length = 15

    # Generate random distances and lengths for the batch
    query_lengths = np.random.randint(low=1, high=max_query_length + 1, size=num_queries)
    reference_lengths = np.random.randint(low=1, high=max_reference_length + 1, size=num_tests)

    query_arrays = [torch.rand(query_len, TEST_EMBED_DIM) for query_len in query_lengths]
    test_arrays = [torch.rand(ref_len, TEST_EMBED_DIM) for ref_len in reference_lengths]

    query_arrays = pad_arrays_or_tensors(query_arrays)
    test_arrays = pad_arrays_or_tensors(test_arrays)

    # Convert similarity to distance
    batched_distances = 1-pairwise_cosine_similarity(query_arrays, test_arrays).numpy()  # type: ignore

    # Compute DTW scores using the function
    dtw_scores = batched_subseq_dtw(batched_distances, query_lengths, reference_lengths)

    # Check that the output shape is correct
    assert dtw_scores.shape == (num_tests,),\
        f"Expected output shape (num_tests,), got {dtw_scores.shape}"

    # Check that the DTW scores are non-negative
    assert np.all(dtw_scores >= 0), "DTW scores should be non-negative"

    # Validate against tslearn's dtw_subsequence_path
    for i in range(num_queries):
        for j in range(num_tests):
            query_len = query_lengths[i]
            ref_len = reference_lengths[j]
            query_array = query_arrays[i][:query_len].numpy()
            test_array = test_arrays[j][:ref_len].numpy()

            # Compute DTW score using tslearn
            _, dtw_score_tslearn = dtw_subsequence_path(query_array, test_array)

            # Compare with our implementation
            assert np.isclose(dtw_scores[j], dtw_score_tslearn, atol=1e-5),\
                f"Mismatch in DTW scores for test {j}: {dtw_scores[j]} vs {dtw_score_tslearn}"