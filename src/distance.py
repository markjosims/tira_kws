from typing import List, Union
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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
