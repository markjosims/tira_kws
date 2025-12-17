import k2
import torch
from typing import *
from constants import DEVICE

"""
## FSA builders
Functions that build FSAs for representing keyword queries
and test phrase distance scores.
"""

def make_dense_fsa(
        distance_scores: torch.Tensor,
        seq_lens: Sequence[int],
) -> k2.DenseFsaVec:
    """
    Creates a dense FSA from a tensor of distance scores.

    Args:
        distance_scores: tensor of distance scores of shape
            T * W_q * W_t, where T is the number of test phrases, W_q
            is the padded sequence length of the query phrase
            and W_t is the padded sequence length of the test phrases
        seq_lens: sequence of un-padded lengths of test phrases
    Returns:
        DenseFsaVec representing keyword

    """
    N = distance_scores.shape[0]

    seq_idcs = torch.arange(N, dtype=torch.int32)
    start_frames = torch.zeros(N, dtype=torch.int32)
    durations = torch.tensor(seq_lens, dtype=torch.int32)

    durations_sorted, indices_sorted = torch.sort(durations, descending=True)
    seq_idcs_sorted = seq_idcs[indices_sorted]
    start_frames_sorted = start_frames[indices_sorted]
    distance_scores_sorted = distance_scores[indices_sorted]

    supervision_segments = torch.stack([seq_idcs_sorted, start_frames_sorted, durations_sorted], dim=1)

    fsa = k2.DenseFsaVec(distance_scores_sorted, supervision_segments).to(DEVICE)
    return fsa

def prepare_dense_fsa_batch(
        prob_matrices: torch.Tensor,
        seq_lens: Sequence[int],
):
    """
    Arguments:
        prob_matrices: tensor of shape Q*T*W_k*W_t, where Q is the number of queries,
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
    query_fsa_str = ""
    for i in range(keyword_len-1):
        curr = str(i)+" "
        nxt = str(i+1)+" "
        #           src_state   dest_state  label   score
        self_arc=   curr +      curr +      curr +  "0.0\n"
        arc=        curr +      nxt +       curr +  "0.0\n"
        query_fsa_str+=self_arc
        query_fsa_str+=arc

    penult = str(keyword_len-1)+" "
    final = str(keyword_len)+" "
    #                   src_state   dest_state  label   score
    final_arc=          penult +    final +     "-1 " + "0.0\n"
    query_fsa_str +=  final_arc
    query_fsa_str += final
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
and test phrase distance scores.
"""

def decode_single_keyword(
        distance_scores: torch.Tensor,
        seq_lens: Sequence[int],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Given a tensor of distance scores of test phrases to a
    query keyphrase, creates an FSA representing the distance
    scores between test and query phrases and another FSA representing
    a path through the keyphrase states. Intersects the two FSAs
    and returns scores and most likely paths for each test phrase
    given the query phrase.

    Args:
        distance_scores: tensor of distance scores of shape
            T * W_t * W_q, where T is the number of test phrases, W_t
            is the padded sequence length of the test phrases
            and W_q is the length of the query phrase
        seq_lens: sequence of un-padded lengths of test phrases

    Returns:
        score, labels: tensor of shape N indicating keyword score
        for each test phrase in batch and tensor of variable shape
        indicating most likely label sequence for intersection of keyword
        and distance scores, with each test phrase label sequence
        separated by a -1 label, which indicates final state.
    """
    dense_fsa = make_dense_fsa(distance_scores, seq_lens)
    keyword_len = distance_scores.shape[-1]
    batch_size = distance_scores.shape[0]
    query_fsa = make_query_fsa(keyword_len, batch_size)

    lattice = k2.intersect_dense(query_fsa, dense_fsa, output_beam=10.0)
    best_path = k2.shortest_path(lattice, use_double_scores=True)
    score = best_path.get_tot_scores(use_double_scores=True, log_semiring=True)
    labels = best_path.labels
    return score, labels

def decode_keyword_batch(
        prob_matrices: torch.Tensor,
        keyword_lens: Sequence[int],
        seq_lens: Sequence[int],
):
    """
    Given a tensor of distance scores of test phrases to a
    batch of query keyphrases, creates an FSA representing the distance
    scores between test and query phrases and another FSA representing
    a path through the keyphrase states. Intersects the two FSAs
    and returns scores and most likely paths for each test phrase
    given the query phrase.

    Arguments:
        prob_matrices: tensor of shape Q*T*W_q*W_t, where Q is the number of queries,
            K the number of test phrases, W_q the number of padded windows in each
            query and W_t the number of padded windows in each test phrase
        keyword_lens: sequence of un-padded lengths of query phrases
        seq_lens: sequence of un-padded lengths of test phrases
    Returns:
        score, labels: tensor of shape Q*T indicating keyword score
        for each test phrase in batch and variable length tensor
        indicating most likely label sequence for intersection of keyword
        and distance scores, with each test phrase label sequence
        separated by a -1 label, which indicates final state.
    """
    num_keywords = prob_matrices.shape[0]
    batch_size = prob_matrices.shape[1]
    query_fsa = prepare_query_graph(keyword_lens, batch_size)
    dense_fsa = prepare_dense_fsa_batch(prob_matrices, seq_lens)

    lattice = k2.intersect_dense(query_fsa, dense_fsa, output_beam=10.0)
    best_path = k2.shortest_path(lattice, use_double_scores=True)
    score = best_path.get_tot_scores(use_double_scores=True, log_semiring=True)
    score = score.reshape(num_keywords, batch_size)
    labels = best_path.labels
    return score, labels