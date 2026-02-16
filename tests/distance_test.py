from tira_kws.distance import get_cosine_similarity, pairwise_cosine_similarity
import torch
import pytest

from tests.test_utils import get_orthogonal_vectors, TEST_EMBED_DIM

@pytest.mark.parametrize(
    "shape", [torch.randint(5, 20, (2,)).tolist() for _ in range(20)]
)
def test_cosine_similarity_matrix_shape(shape):
    n_records_a, n_records_b = shape
    emb_a = torch.randn(n_records_a, TEST_EMBED_DIM)
    emb_b = torch.randn(n_records_b, TEST_EMBED_DIM)
    sim_matrix = get_cosine_similarity(emb_a, emb_b)
    assert sim_matrix.shape == (n_records_a, n_records_b)

@pytest.mark.parametrize(
    "n_vectors", [torch.randint(5, 20, (1,)).item() for _ in range(20)]
)
def test_cosine_similarity_values(n_vectors):
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=n_vectors)
    matrix_a = get_cosine_similarity(emb_a, emb_a)
    matrix_b = get_cosine_similarity(emb_b, emb_b)
    matrix_ab = get_cosine_similarity(emb_a, emb_b)

    assert matrix_a.mean() > matrix_ab.mean()
    assert matrix_b.mean() > matrix_ab.mean()

@pytest.mark.parametrize(
    "shape", [torch.randint(5, 20, (4,)).tolist() for _ in range(20)]
)
def test_pairwise_similarity_shape(shape):
    n_records_a, n_windows_a, n_records_b, n_windows_b = shape

    emb_a = torch.randn(n_records_a, n_windows_a, TEST_EMBED_DIM)
    emb_b = torch.randn(n_records_b, n_windows_b, TEST_EMBED_DIM)

    pairwise_similarity = pairwise_cosine_similarity(emb_a, emb_b)

    assert pairwise_similarity.shape == (n_records_a, n_records_b, n_windows_a, n_windows_b)

@pytest.mark.parametrize(
    "n_windows", [torch.randint(5, 20, (2,)).tolist() for _ in range(20)]
)
def test_pairwise_similarity_values(n_windows):
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=10, n_windows=n_windows)

    pairwise_similarity = pairwise_cosine_similarity(emb_a, emb_b)

    for i in range(pairwise_similarity.shape[0]):
        for j in range(pairwise_similarity.shape[1]):
            record_sim_pred = pairwise_similarity[i, j]
            record_sim_expected = get_cosine_similarity(emb_a[i], emb_b[j])
            assert torch.isclose(record_sim_pred, record_sim_expected, atol=1e-06).all()

