from src.dtw import batched_subseq_dtw, dtw_subsequence_path
from src.distance import pairwise_cosine_similarity, pad_arrays_or_tensors
from tests.test_utils import TEST_EMBED_DIM
import time
import pytest
from random import randint
import numpy as np
import torch

@pytest.mark.parametrize(
    "num_queries,num_tests", [(randint(10, 50), randint(10, 50)) for _ in range(60)]
)
def test_batched_subseq_dtw(num_queries, num_tests):
    max_query_length = 20
    max_reference_length = 30
    query_lengths, reference_lengths, query_arrays, test_arrays, \
    batched_distances, query_lengths_expanded, reference_lengths_expanded \
    = create_dtw_test_inputs(
        num_queries,
        num_tests,
        max_query_length,
        max_reference_length,
        TEST_EMBED_DIM
    )

    # Compute DTW scores using the function
    start_time = time.perf_counter()
    dtw_scores = batched_subseq_dtw(
        batched_distances,
        query_lengths_expanded,
        reference_lengths_expanded
    )
    end_time = time.perf_counter()
    print(f"DTW computation time: {end_time - start_time:.4f} seconds")

    # Check that the output shape is correct
    assert dtw_scores.shape == (num_queries * num_tests,),\
        f"Expected output shape (num_queries * num_tests,), got {dtw_scores.shape}"

    # Check that the DTW scores are non-negative
    assert np.all(dtw_scores >= 0), "DTW scores should be non-negative"

    tslearn_time = 0
    # Validate against tslearn's dtw_subsequence_path
    for i in range(num_queries):
        for j in range(num_tests):
            query_len = query_lengths[i]
            ref_len = reference_lengths[j]
            query_array = query_arrays[i][:query_len].numpy()
            test_array = test_arrays[j][:ref_len].numpy()

            # Compute DTW score using tslearn
            start_time = time.perf_counter()
            _, dtw_score_tslearn = dtw_subsequence_path(query_array, test_array)
            end_time = time.perf_counter()
            tslearn_time += end_time - start_time

            # Compare with our implementation
            idx = i * num_tests + j
            assert np.isclose(dtw_scores[idx], dtw_score_tslearn, atol=1e-5),\
                f"Mismatch in DTW scores for test {j}: {dtw_scores[idx]} vs {dtw_score_tslearn}"
    print(f"Total tslearn DTW computation time: {tslearn_time:.4f} seconds")
    print(f"Native implementation is {tslearn_time / (end_time - start_time):.2f} times faster than tslearn")

@pytest.mark.parametrize(
    "num_queries,num_tests", [(randint(10, 50), randint(10, 50)) for _ in range(60)]
)
def test_batched_subseq_dtw_time(num_queries, num_tests):
    max_query_length = 50
    max_reference_length = 100
    query_lengths, reference_lengths, query_arrays, test_arrays, \
    batched_distances, query_lengths_expanded, reference_lengths_expanded \
    = create_dtw_test_inputs(
        num_queries,
        num_tests,
        max_query_length,
        max_reference_length,
        TEST_EMBED_DIM*10
    )

    # Compute DTW scores w/o prange
    start_time = time.perf_counter()
    dtw_scores_no_prange = batched_subseq_dtw(
        batched_distances,
        query_lengths_expanded,
        reference_lengths_expanded,
        use_prange=False
    )
    end_time = time.perf_counter()
    no_prange_time = end_time - start_time
    print(f"DTW computation time without prange: {no_prange_time:.4f} seconds")  

    # Compute DTW scores with prange
    start_time = time.perf_counter()
    dtw_scores_prange = batched_subseq_dtw(
        batched_distances,
        query_lengths_expanded,
        reference_lengths_expanded,
        use_prange=True
    )
    end_time = time.perf_counter()
    prange_time = end_time - start_time
    print(f"DTW computation time with prange: {prange_time:.4f} seconds")

    print(f"Speedup with prange: {no_prange_time / prange_time:.2f}x")

def create_dtw_test_inputs(num_queries, num_tests, max_query_length, max_reference_length, embed_dim):

    # Generate random distances and lengths for the batch
    query_lengths = np.random.randint(low=1, high=max_query_length + 1, size=num_queries)
    reference_lengths = np.random.randint(low=max_query_length, high=max_reference_length + 1, size=num_tests)

    query_array_list = [torch.rand(query_len, embed_dim) for query_len in query_lengths]
    test_array_list = [torch.rand(ref_len, embed_dim) for ref_len in reference_lengths]

    query_arrays = pad_arrays_or_tensors(query_array_list)
    test_arrays = pad_arrays_or_tensors(test_array_list)
    # Convert similarity to distance
    batched_distances = 1-pairwise_cosine_similarity(query_arrays, test_arrays) # type: ignore

    # Flatten the distance matrices to shape
    # (num_queries*num_tests, max_query_length, max_reference_length)
    batched_distances = batched_distances.flatten(start_dim=0, end_dim=1)

    # Convert to numpy array for the DTW function
    batched_distances = batched_distances.numpy()

    # reshape query_lengths and reference_lengths to match the batch size
    query_lengths_expanded = np.repeat(query_lengths, num_tests)
    reference_lengths_expanded = np.tile(reference_lengths, num_queries)
    return query_lengths,reference_lengths,query_arrays,test_arrays,batched_distances,query_lengths_expanded,reference_lengths_expanded