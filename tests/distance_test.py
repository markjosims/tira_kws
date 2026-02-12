from sqlalchemy.orm import query_expression

from distance import get_cosine_distance, get_cosine_similarity, get_windowed_cosine_similarity
import torch
import pytest
from src.wfst import decode_keyword_batch, decode_single_keyword, decode_embed_list
from random import randint
from tslearn.metrics import dtw_path_from_metric
from typing import Optional, Tuple, List

TEST_EMBED_DIM = 256

def get_orthogonal_vectors(
        n_vectors: int = 8,
        n_windows: Optional[Tuple[int, int]]=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_vector = torch.randn(1, TEST_EMBED_DIM)
    ortho_vector = torch.linalg.svd(base_vector).Vh[1]
    scaling_factor = 0.01
    noise_a = torch.randn(n_vectors, TEST_EMBED_DIM) * scaling_factor
    noise_b = torch.randn(n_vectors, TEST_EMBED_DIM) * scaling_factor

    vectors_a = base_vector + noise_a
    vectors_b = ortho_vector + noise_b

    if n_windows is not None:
        windows_a, windows_b = n_windows
        vectors_a = torch.stack([vectors_a]*windows_a, dim=1)
        vectors_b = torch.stack([vectors_b]*windows_b, dim=1)

        noise_a = torch.randn(n_vectors, windows_a, TEST_EMBED_DIM) * scaling_factor
        noise_b = torch.randn(n_vectors, windows_b, TEST_EMBED_DIM) * scaling_factor

        vectors_a += noise_a
        vectors_b += noise_b

    return vectors_a, vectors_b

def noise_pad(t: torch.Tensor, n_pad: int = 20) -> torch.Tensor:
    scaling_factor = 200
    noise = torch.full((n_pad, t.shape[-1]), torch.inf) * scaling_factor
    return torch.cat([noise, t, noise], dim=0)

@pytest.mark.parametrize(
    "shape", [torch.randint(5, 20, (2,)).tolist() for _ in range(20)]
)
def test_cosine_similarity_matrix_shape(shape):
    n_records_a, n_records_b = shape
    emb_a = torch.randn(5, TEST_EMBED_DIM)
    emb_b = torch.randn(3, TEST_EMBED_DIM)
    sim_matrix = get_cosine_similarity(emb_a, emb_b)
    assert sim_matrix.shape == (5, 3)

@pytest.mark.parametrize(
    "n_vectors", [torch.randint(5, 20, (1,)).item() for _ in range(20)]
)
def test_cosine_similarity_values(n_vectors):
    emb_a, emb_b = get_orthogonal_vectors()
    matrix_a = get_cosine_similarity(emb_a, emb_a)
    matrix_b = get_cosine_similarity(emb_b, emb_b)
    matrix_ab = get_cosine_similarity(emb_a, emb_b)

    assert matrix_a.mean() > matrix_ab.mean()
    assert matrix_b.mean() > matrix_ab.mean()

@pytest.mark.parametrize(
    "shape", [torch.randint(5, 20, (4,)).tolist() for _ in range(20)]
)
def test_windowed_similarity_shape(shape):
    n_records_a, n_windows_a, n_records_b, n_windows_b = shape

    emb_a = torch.randn(n_records_a, n_windows_a, TEST_EMBED_DIM)
    emb_b = torch.randn(n_records_b, n_windows_b, TEST_EMBED_DIM)

    windowed_similarity = get_windowed_cosine_similarity(emb_a, emb_b)

    assert windowed_similarity.shape == (n_records_a, n_records_b, n_windows_a, n_windows_b)

@pytest.mark.parametrize(
    "n_windows", [torch.randint(5, 20, (2,)).tolist() for _ in range(20)]
)
def test_windowed_similarity_values(n_windows):
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=10, n_windows=n_windows)

    windowed_similarity = get_windowed_cosine_similarity(emb_a, emb_b)

    for i in range(windowed_similarity.shape[0]):
        for j in range(windowed_similarity.shape[1]):
            record_sim_pred = windowed_similarity[i, j]
            record_sim_expected = get_cosine_similarity(emb_a[i], emb_b[j])
            assert torch.isclose(record_sim_pred, record_sim_expected, atol=1e-06).all()

@pytest.mark.parametrize(
    "batch_size,n_windows", [(randint(5, 20),randint(5, 20)) for _ in range(60)]
)
def test_decode_single_wfst(batch_size, n_windows):
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=batch_size, n_windows=(n_windows,n_windows))
    emb_a_trunc = emb_a[:,:n_windows//2,:]
    # emb_a_stacked = torch.concat([emb_a.clone(), emb_b])
    windowed_similarity = get_windowed_cosine_similarity(emb_a_trunc, emb_a)
    # `windowed_similarity` is shape Q*T*W_q*W_t
    # get T*W_q*W_t tensor indicating windowed hit probabilities for first keyword
    first_query_similarity = windowed_similarity[0]
    seq_lens = [randint(1, n_windows-1) for _ in range(batch_size)]
    # `decode_single_keyword` expects a tensor of shape T*W_t*W_q
    # need to swap last two dimensions of `first_query_similarity`
    first_query_similarity = first_query_similarity.transpose(1,2)
    scores, labels = decode_single_keyword(first_query_similarity, seq_lens)
    assert scores.shape == (batch_size,)

@pytest.mark.parametrize(
    "query_windows", [randint(5, 20) for _ in range(60)]
)
def test_wfst_sim(query_windows):
    n_windows = [query_windows*2, query_windows*2]
    batch_size = 16
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=batch_size, n_windows=n_windows)
    query_embeds = emb_a[:,:query_windows,:]
    test_embeds = torch.concat([emb_a, emb_b])

    windowed_similarity = get_windowed_cosine_similarity(query_embeds, test_embeds)
    keyword_lens = torch.full((batch_size,), query_windows, dtype=torch.int64)
    seq_lens = torch.full((batch_size*2,), query_windows*2, dtype=torch.int64)

    wfst_scores, labels = decode_keyword_batch(windowed_similarity, keyword_lens, seq_lens)
    assert wfst_scores.shape == (batch_size, batch_size*2)
    assert wfst_scores.sum() != -torch.inf
    assert wfst_scores.sum() != torch.inf

    for i in range(batch_size):
        for j in range(batch_size):
            self_score = wfst_scores[i,j]
            cross_score = wfst_scores[i,j+batch_size]
            assert self_score > cross_score

@pytest.mark.parametrize(
    "n_windows", [[randint(5, 20)]*2 for _ in range(60)]
)
def test_wfst_sim_same_len(n_windows):
    """
    `decode_keyword_batch` needs to be able to handle inputs
    where the test phrases have the same shape as the query phrases
    """
    batch_size = 16
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=batch_size, n_windows=n_windows)

    self_similarity = get_windowed_cosine_similarity(emb_a, emb_a)
    cross_similarity = get_windowed_cosine_similarity(emb_a, emb_b)
    keyword_lens = torch.full((batch_size,), n_windows[0], dtype=torch.int64)
    seq_lens = torch.full((batch_size,), n_windows[1], dtype=torch.int64)

    self_scores, self_labels = decode_keyword_batch(self_similarity, keyword_lens, seq_lens)
    cross_scores, cross_labels = decode_keyword_batch(cross_similarity, keyword_lens, seq_lens)
    assert self_scores.shape == (batch_size, batch_size)
    assert cross_scores.shape == (batch_size, batch_size)

    assert self_scores.sum() != -torch.inf
    assert cross_scores.sum() != -torch.inf

    for i in range(batch_size):
        for j in range(batch_size):
            assert self_scores[i,j] > cross_scores[i,j]

@pytest.mark.parametrize(
    "query_windows", [randint(5, 20) for _ in range(60)]
)
def test_wfst_sim_variable_len(query_windows):
    test_windows = query_windows*2
    n_windows = (test_windows, test_windows)
    batch_size = 16
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=batch_size, n_windows=n_windows)
    query_embeds = emb_a[:,:query_windows,:].clone()
    test_embeds = torch.concat([emb_a, emb_b])

    windowed_similarity = get_windowed_cosine_similarity(query_embeds, test_embeds)
    keyword_lens = [randint(1, query_windows) for _ in range(batch_size)]
    seq_lens = [randint(query_windows, test_windows) for _ in range(batch_size*2)]

    wfst_scores, labels = decode_keyword_batch(windowed_similarity, keyword_lens, seq_lens)
    assert wfst_scores.shape == (batch_size, batch_size*2)
    assert wfst_scores.sum() != -torch.inf
    assert wfst_scores.sum() != torch.inf

@pytest.mark.parametrize(
    "num_queries,num_tests", [(randint(5, 20), randint(5,20)) for _ in range(60)]
)
def test_decode_embed_list(num_queries, num_tests):
    query_embed_lens = [randint(5, 20) for _ in range(num_queries)]
    test_embed_lens = [randint(21, 30) for _ in range(num_tests)]
    query_embeds = [torch.rand(l, TEST_EMBED_DIM) for l in query_embed_lens]
    test_embeds = [torch.rand(l, TEST_EMBED_DIM) for l in test_embed_lens]

    scores, labels = decode_embed_list(query_embeds, test_embeds)
    assert scores.shape == (num_queries, num_tests)
    assert scores.sum() != -torch.inf
    assert scores.sum() != torch.inf