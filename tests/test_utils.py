import torch
from typing import Optional, Tuple

TEST_EMBED_DIM = 32

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
