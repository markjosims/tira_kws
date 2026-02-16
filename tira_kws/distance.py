from typing import List, Union, Tuple
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from tira_kws.constants import DEVICE

"""
## sequence utilities
- pad_arrays_or_tensors: Pad a list of variable-length sequences to a common length.
"""

def pad_arrays_or_tensors(
        sequences: Union[List[torch.Tensor], List[np.ndarray]],
        padding_value: float = float('inf'),
    ) -> Union[torch.Tensor, np.ndarray]:
    """
    Pad a sequence of embeddings to a common length. Encapsulates logic
    for both torch and numpy tensors.
    """
    if isinstance(sequences[0], torch.Tensor):
        return pad_sequence(sequences, batch_first=True, padding_value=padding_value) # type: ignore
    elif isinstance(sequences[0], np.ndarray):
        max_length = max(seq.shape[0] for seq in sequences)
        padded_sequences = np.full((len(sequences), max_length, sequences[0].shape[1]), padding_value)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :seq.shape[0], :] = seq
        return padded_sequences
    else:
        raise ValueError("Unsupported sequence type. Expected torch.Tensor or np.ndarray.")
    
def pad_matrices(
        matrices: Union[List[torch.Tensor], List[np.ndarray]],
        padding_value: float = float('inf'),
) -> Union[torch.Tensor, np.ndarray]:
    """
    Pad a sequence of matrices to a common shape. Encapsulates logic
    for both torch and numpy tensors.
    """
    if isinstance(matrices[0], torch.Tensor):
        max_rows = max(mat.shape[0] for mat in matrices)
        max_cols = max(mat.shape[1] for mat in matrices)
        padded_matrices = torch.full((len(matrices), max_rows, max_cols), padding_value)
        for i, mat in enumerate(matrices):
            padded_matrices[i, :mat.shape[0], :mat.shape[1]] = mat
        return padded_matrices
    elif isinstance(matrices[0], np.ndarray):
        max_rows = max(mat.shape[0] for mat in matrices)
        max_cols = max(mat.shape[1] for mat in matrices)
        padded_matrices = np.full((len(matrices), max_rows, max_cols), padding_value)
        for i, mat in enumerate(matrices):
            padded_matrices[i, :mat.shape[0], :mat.shape[1]] = mat
        return padded_matrices
    else:
        raise ValueError("Unsupported matrix type. Expected torch.Tensor or np.ndarray.")

def pad_and_return_lengths(
        batch_embeds: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        batch_embeds: list of torch.Tensors indicating embedding tensors
            for current batch

    Returns: (padded_tensor,seq_lens) torch.Tensor containing padded
        embeddings for the current batch and a torch.Tensor containing
        the unpadded length of each sequence in the batch
    """

    # # enforce pad length for whole sequence by zero-padding first element
    # first_row_len = batch_embeds[0].shape[0]
    # embed_dim = batch_embeds[0].shape[-1]
    # first_row_padding = torch.zeros(pad_len-first_row_len, embed_dim)
    # first_row_padded = torch.cat([batch_embeds[0], first_row_padding], dim=0)
    # batch_embeds[0] = first_row_padded

    seq_lens = torch.tensor(
        [seq.shape[0] for seq in batch_embeds],
        dtype=int,
        device=DEVICE,
    )
    padded_batch = pad_sequence(
        batch_embeds,
        batch_first=True,
        padding_value=0.0,

    )
    padded_batch.to(DEVICE)
    return padded_batch, seq_lens

"""
## similarity computation utilities
- get_cosine_similarity: Compute cosine similarity matrix between two sets of embeddings.
- get_cosine_distance: Compute cosine distance matrix between two sets of embeddings.
- get_windowed_cosine_similarity: Compute cosine similarity matrix between windowed embeddings.
"""

# @torch.compile
def get_cosine_similarity(
        query_embeds: torch.Tensor,
        test_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity scores between query embeddings and test embeddings
    where the (i,j)^th element is the cosine similarity between the i^th query
    embedding and the j^th test embedding.

    Args:
        query_embeds: Tensor of shape (num_queries, embed_dim)
        test_embeds: Tensor of shape (num_tests, embed_dim)
    Returns:
        Tensor of shape (num_queries, num_tests) with cosine similarity scores
    """
    query_norm = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
    test_norm = test_embeds / test_embeds.norm(dim=-1, keepdim=True)
    similarity_scores = torch.matmul(query_norm, test_norm.T)
    return similarity_scores

def get_cosine_distance(
        query_embeds: torch.Tensor,
        test_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Computes cosine distance, i.e. 1-cosine similarity,
    between query embeddings and test embeddings

    Args:
        query_embeds: Tensor of shape (num_queries, embed_dim)
        test_embeds: Tensor of shape (num_tests, embed_dim)
    Returns:
        Tensor of shape (num_queries, num_tests) with cosine distance scores
    """
    return 1-get_cosine_similarity(query_embeds, test_embeds)

# @torch.compile
def pairwise_cosine_similarity(
        query_embeds: torch.Tensor,
        test_embeds: torch.Tensor,
        fillna: bool = False,
) -> torch.Tensor:
    """
    Computes cosine similarity scores between windowed query embeddings and test embeddings.
    Returns a 4-d tensor where each (i,j)^th element is a similarity matrix between
    the windowed embeddings from the i^th query and j^th test phrase.

    Args:
        query_embeds: Tensor of shape (num_queries, num_windows, embed_dim)
        test_embeds: Tensor of shape (num_tests, num_windows, embed_dim)
        fillna: (bool) indicates whether to fill nan values in the resulting
            similarity tensor. If True, fills nan values with 2.0 (i.e. maximum
            similarity)

    Returns:
        Tensor of shape (num_queries, num_tests, num_windows_query, num_windows_test)
    """
    query_norm = query_embeds / query_embeds.norm(dim=-1, p=2, keepdim=True)
    test_norm = test_embeds / test_embeds.norm(dim=-1, p=2, keepdim=True)
    # k = keyword index,    i = keyword window index,   d = embedding dim
    # t = test index,       j = test window index,      d = embedding dim
    # scores = torch.einsum('kid,tjd->ktij', query_norm, test_norm)
    query_expanded = query_norm.unsqueeze(1)
    test_expanded = test_norm.transpose(-1, -2).unsqueeze(0)
    similarity_scores = query_expanded @ test_expanded

    if fillna:
        similarity_scores = similarity_scores.nan_to_num(-1.0)

    return similarity_scores
