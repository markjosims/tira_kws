from tqdm import tqdm

from encoding import *
import librosa
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor
from constants import DEVICE, SAMPLE_RATE
import pytest
from tslearn.metrics import dtw_subsequence_path
from wfst import decode_keyword_batch

AILN_WAV = "data/ailn.wav"
EXPECTED_CLAP_IPA_SIZE = {
    'tiny': 384,
    'base': 512,
    'small': 512,
}
TEST_EMBED_DIM = 256

def get_orthogonal_vectors(
        n_vectors: int = 8,
        n_windows: Optional[tuple[int, int]]=None
) -> tuple[torch.Tensor, torch.Tensor]:
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

def load_test_audio():
    audio, _ = librosa.load(AILN_WAV, sr=SAMPLE_RATE)
    audio = np.expand_dims(audio, axis=0)
    return audio

def test_load_clap_speech_encoder():
    for size in ['tiny', 'base', 'small']:
        encoder = load_clap_speech_encoder(size)
        assert isinstance(encoder, SpeechEncoder)
        assert encoder.device == DEVICE

def test_load_clap_speech_processor():
    processor = load_clap_speech_processor()
    assert isinstance(processor, WhisperProcessor)

def test_encode_clap_audio():
    audio = load_test_audio()
    processor = load_clap_speech_processor()
    for size in ['tiny', 'base', 'small']:
        encoder = load_clap_speech_encoder(size)
        embeddings = encode_clap_audio(audio, encoder, processor)
        assert embeddings.shape[0] == audio.shape[0]
        assert embeddings.shape[1] == EXPECTED_CLAP_IPA_SIZE[size]

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
    "n_windows", [[torch.randint(5, 20, (1,)).item()]*2 for _ in range(20)]
)
def test_wfst_sim(n_windows):
    batch_size = 16
    emb_a, emb_b = get_orthogonal_vectors(n_vectors=batch_size, n_windows=n_windows)
    query_embeds = emb_a
    test_embeds = torch.concat([emb_a, emb_b])

    windowed_similarity = get_windowed_cosine_similarity(query_embeds, test_embeds)
    keyword_lens = torch.full((batch_size,), n_windows[0], dtype=torch.int64)
    seq_lens = torch.full((batch_size,), n_windows[1], dtype=torch.int64)

    wfst_scores = decode_keyword_batch(windowed_similarity, keyword_lens, seq_lens)
    assert wfst_scores.shape == (batch_size, batch_size*2)

    for i in range(batch_size):
        for j in range(batch_size):
            in_domain_score = wfst_scores[i,j]
            out_domain_score = wfst_scores[i,j+batch_size]
            assert in_domain_score < out_domain_score
