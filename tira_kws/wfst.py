from __future__ import annotations

from typing import List, Literal, Tuple

import k2
import torch
from typing import *
from torch.nn.utils.rnn import pad_sequence
from tira_kws.constants import DEVICE, WFST_BATCH_SIZE
from tira_kws.distance import pairwise_cosine_similarity, pad_and_return_lengths
from torch.utils.data import DataLoader

SELF_WEIGHT = torch.tensor(1).log()
SKIP_WEIGHT = torch.tensor(1).log()

"""
## FSA builders
Functions that build FSAs for representing keyword queries
and test phrase similarity scores.
"""

def make_dense_fsa(
        similarity_scores: torch.Tensor,
        seq_lens: Sequence[int],
) -> k2.DenseFsaVec:
    """
    Creates a dense FSA from a tensor of similarity scores.

    Args:
        similarity_scores: tensor of similarity scores of shape
            T * W_t * W_q, where T is the number of test phrases, W_t
            is the padded sequence length of the test phrases
            and W_q is the padded sequence length of the query phrases
        seq_lens: sequence of un-padded lengths of test phrases
    Returns:
        DenseFsaVec representing keyword

    """
    N = similarity_scores.shape[0]

    # similarity scores have range [-1,1]
    # rescale to [0,1] then take log
    similarity_scores_rescaled = (1+similarity_scores)/2 + 1e-8
    similarity_scores_rescaled = similarity_scores_rescaled.log()

    seq_idcs = torch.arange(N, dtype=torch.int32)
    start_frames = torch.zeros(N, dtype=torch.int32)
    durations = torch.tensor(seq_lens, dtype=torch.int32)

    durations_sorted, indices_sorted = torch.sort(durations, descending=True)
    seq_idcs_sorted = seq_idcs[indices_sorted]
    start_frames_sorted = start_frames[indices_sorted]
    similarity_scores_sorted = similarity_scores_rescaled[indices_sorted]

    # append column to end representing average probability across
    # all query frames
    mean_similarity = similarity_scores_sorted.mean(dim=-1, keepdim=True)
    similarity_scores_w_mean = torch.concat([similarity_scores_sorted, mean_similarity], dim=-1)

    supervision_segments = torch.stack([seq_idcs_sorted, start_frames_sorted, durations_sorted], dim=1)

    fsa = k2.DenseFsaVec(similarity_scores_w_mean, supervision_segments).to(DEVICE)
    return fsa

def prepare_dense_fsa_batch(
        prob_matrices: torch.Tensor,
        seq_lens: Sequence[int],
):
    """
    Arguments:
        prob_matrices: tensor of shape Q*T*W_q*W_t, where Q is the number of queries,
            T the number of test phrases, W_q the number of padded windows in each
            query and W_t the number of padded windows in each test phrase
        seq_lens: sequence of un-padded lengths of test phrases
    Returns:
        Dense FSA of query probabilities
    """
    num_keywords = prob_matrices.shape[0]
    # prob_matrices needs to be reshaped from K*T*W_k*W_t
    # to (K*T)*W_t*W_k
    probs_flattened = prob_matrices.flatten(0,1)
    probs_transposed = probs_flattened.transpose(1,2)
    probs_transposed = probs_transposed.to(DEVICE)
    if type(seq_lens) is torch.Tensor:
        seq_lens = seq_lens.tolist()
    new_seq_lens=seq_lens*num_keywords
    return make_dense_fsa(probs_transposed, new_seq_lens)

def get_query_fsa_str(keyword_len: int) -> str:
    """
    Helper function for `make_query_fsa` that generates a string
    interpretable by `k2` as an FSA.
    Args:
        keyword_len: int indicating number of states in keyword query
    Returns:
        query_fsa_str: string representation of FSA accepting keyword labels

    """
    if type(keyword_len) is torch.Tensor:
        keyword_len = keyword_len.item()

    arc_template = "{curr_state} {next_state} {label} {score}\n"
    arc_list = []
    initial_self_arc = arc_template.format(
        curr_state=0,
        next_state=0,
        label=keyword_len,
        score=0.0,
    )
    arc_list.append(initial_self_arc)

    for i in range(keyword_len-1):
        curr = str(i)+" "
        nxt = str(i+1)+" "
        arc = arc_template.format(
            curr_state=curr,
            next_state=nxt,
            label=curr,
            score=0.0,
        )
        arc_list.append(arc)
        self_arc = arc_template.format(
            curr_state=curr,
            next_state=nxt,
            label=curr,#keyword_len,
            score=SELF_WEIGHT,
        )
        arc_list.append(self_arc)
        if i < keyword_len-2:
            skip_arc = arc_template.format(
                curr_state=curr,
                next_state=nxt,
                label=keyword_len,
                score=SKIP_WEIGHT,
            )
            arc_list.append(skip_arc)


    final_self_arc = arc_template.format(
        curr_state=keyword_len-1,
        next_state=keyword_len-1,
        label=keyword_len,
        score=0.0,
    )
    final_arc = arc_template.format(
        curr_state=keyword_len-1,
        next_state=keyword_len,
        label=-1,
        score=0.0,
    )
    final_state = f"{keyword_len}"

    arc_list.append(final_self_arc)
    arc_list.append(final_arc)
    arc_list.append(final_state)

    query_fsa_str = "\n".join(arc_list)
    return query_fsa_str

def make_query_fsa(keyword_len, batch_size: Optional[int]=None):
    """
    Creates an FSA that accepts any contiguous monotonic sequence of labels from 0
    to keyword_len-1, allowing labels to be repeated.

    Args:
        keyword_len: int indicating number of states in keyword query
        batch_size: Optional[int] indicating expected batch size of test phrases

    Returns:
        FSA with forward and self-loops monotonically traversing states 0
        to keyword_len-1
    """
    query_fsa_str = get_query_fsa_str(keyword_len)
    fsa = k2.Fsa.from_str(query_fsa_str).to(DEVICE)

    if batch_size is not None:
        fsa = k2.create_fsa_vec([fsa]*batch_size)
    return fsa

def prepare_query_graph(keyword_lens, batch_size):
    """
    Creates an FSA Vector with `batch_size` repetitions
    of each query graph.

    Arguments:
        keyword_lens: list of lengths of each query graph
        batch_size: number of test phrases in batch
    Returns:
        FSA vector of queries
    """
    expanded_lens = torch.tensor(keyword_lens) \
        .unsqueeze(1) \
        .repeat(1, batch_size) \
        .view(-1) \
        .to(torch.int32)
    fsa_list = []
    for keyword_len in expanded_lens:
        fsa_list.append(make_query_fsa(keyword_len))
    query_graph = k2.create_fsa_vec(fsa_list)
    return query_graph

"""
## Decoding functions
Functions that build and decode FSAs for representing keyword queries
and test phrase similarity scores.
"""

def decode_single_keyword(
        similarity_scores: torch.Tensor,
        seq_lens: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor of similarity scores of test phrases to a
    query keyphrase, creates an FSA representing the similarity
    scores between test and query phrases and another FSA representing
    a path through the keyphrase states. Intersects the two FSAs
    and returns scores and most likely paths for each test phrase
    given the query phrase.

    Args:
        similarity_scores: tensor of similarity scores of shape
            T * W_t * W_q, where T is the number of test phrases, W_t
            is the padded sequence length of the test phrases
            and W_q is the length of the query phrase
        seq_lens: sequence of un-padded lengths of test phrases

    Returns:
        score, labels: tensor of shape N indicating keyword score
        for each test phrase in batch and tensor of variable shape
        indicating most likely label sequence for intersection of keyword
        and similarity scores, with each test phrase label sequence
        separated by a -1 label, which indicates final state.
    """
    dense_fsa = make_dense_fsa(similarity_scores, seq_lens)
    keyword_len = similarity_scores.shape[-1]
    batch_size = similarity_scores.shape[0]
    query_fsa = make_query_fsa(keyword_len, batch_size)

    lattice = k2.intersect_dense(query_fsa, dense_fsa, output_beam=10.0)
    best_path = k2.shortest_path(lattice, use_double_scores=True)
    score = best_path.get_tot_scores(use_double_scores=True, log_semiring=True)
    labels = best_path.labels
    return score, labels

def decode_keyword_batch(
        similarity_tensor: torch.Tensor,
        keyword_lens: Sequence[int],
        seq_lens: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor of similarity scores of test phrases to a
    batch of query keyphrases, creates an FSA representing the similarity
    scores between test and query phrases and another FSA representing
    a path through the keyphrase states. Intersects the two FSAs
    and returns scores and most likely paths for each test phrase
    given the query phrase.

    Arguments:
        similarity_tensor: tensor of shape Q*T*W_q*W_t, where Q is the number of queries,
            K the number of test phrases, W_q the number of padded windows in each
            query and W_t the number of padded windows in each test phrase
        keyword_lens: sequence of un-padded lengths of query phrases
        seq_lens: sequence of un-padded lengths of test phrases
    Returns:
        score, labels: tensor of shape Q*T indicating keyword score
        for each test phrase in batch and variable length tensor
        indicating most likely label sequence for intersection of keyword
        and similarity scores, with each test phrase label sequence
        separated by a -1 label, which indicates final state.
    """
    num_keywords = similarity_tensor.shape[0]
    batch_size = similarity_tensor.shape[1]
    query_fsa = prepare_query_graph(keyword_lens, batch_size)
    dense_fsa = prepare_dense_fsa_batch(similarity_tensor, seq_lens)

    lattice = k2.intersect_dense(query_fsa, dense_fsa, output_beam=10.0)
    best_path = k2.shortest_path(lattice, use_double_scores=True)
    score = best_path.get_tot_scores(use_double_scores=True, log_semiring=True)
    score = score.reshape(num_keywords, batch_size)
    labels = best_path.labels

    # normalize scores by test phrase lens
    score = score / torch.tensor(seq_lens).unsqueeze(0)

    return score, labels

def decode_embed_list(
        query_embeds: List[torch.Tensor],
        test_embeds: List[torch.Tensor],
        similarity_metric: str = 'cosine',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decodes a list of query and test phrase embeddings, handling batching
    and padding.

    Args:
        query_embeds: List of embeddings for query keyphrases
        test_embeds: List of embeddings for test phrases
        similarity_metric: Type of similarity metric to use.
            For now only 'cosine' is supported

    Returns: (scores, labels): torch.Tensors containing keyword hit scores
        and shortest path labels for each test phrase in the batch
    """
    if similarity_metric == 'cosine':
        similarity_function = lambda *args: pairwise_cosine_similarity(*args, fillna=True)
    else:
        raise ValueError(f'Unknown similarity metric: {similarity_metric}')

    # get keyword lengths and pad
    keyword_lens = [query.shape[0] for query in query_embeds]
    query_embeds_padded = pad_sequence(query_embeds, batch_first=True, padding_value=0.0)
    query_embeds_padded.to(DEVICE)

    # test phrase padding is handled by DataLoader
    dataloader = DataLoader(
        test_embeds,
        collate_fn=pad_and_return_lengths,
        batch_size=WFST_BATCH_SIZE,
    )

    score_list = []
    label_list = []
    for batch_test_embeds, batch_seq_lens in dataloader:
        batch_similarity = similarity_function(query_embeds_padded, batch_test_embeds)
        batch_scores, batch_labels = decode_keyword_batch(
            batch_similarity, keyword_lens, batch_seq_lens
        )
        score_list.append(batch_scores)
        label_list.append(batch_labels)

    scores = torch.cat(score_list, dim=1)
    labels = torch.cat(label_list, dim=0)
    return scores, labels