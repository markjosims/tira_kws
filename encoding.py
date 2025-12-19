from __future__ import annotations
from typing import *
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from argparse import ArgumentParser
from constants import (
    CLAP_IS_AVAILABLE, SPEECHBRAIN_IS_AVAILABLE, DEVICE, SAMPLE_RATE,
    SPEECHBRAIN_LID_ENCODER_NAME, CLAP_IPA_ENCODER_NAME,
)

if CLAP_IS_AVAILABLE or TYPE_CHECKING:
    from clap.encoders import SpeechEncoder, PhoneEncoder
if SPEECHBRAIN_IS_AVAILABLE or TYPE_CHECKING:
    from speechbrain.inference.classifiers import EncoderClassifier

"""
## Speech encoding utilities
- add_sliding_window_args: Add command-line arguments for sliding window parameters.
- add_encoder_args: Add command-line arguments for encoder model selection.
- get_sliding_window: Split audio into overlapping windows for frame-level processing.
- get_frame: Helper function for `get_sliding_window` to extract a specific frame from an audio sample.
"""

def add_sliding_window_args(parser):
    parser.add_argument('--window_size', '-w', type=float, default=None, help='Window size in seconds for sliding window embedding extraction.')
    parser.add_argument('--window_hop', '-p', type=float, default=None, help='Window hop in seconds for sliding window embedding extraction.')
    return parser

def add_encoder_args(parser: ArgumentParser):
    parser.add_argument(
        '--encoder', '-e', type=str, default='clap_ipa', choices=['clap_ipa', 'speechbrain_lid'],
        help='Encoder model to use for generating embeddings.'
    )
    parser.add_argument('--encoder_size', '-s', type=str, default='base', help='Size of CLAP speech encoder to use.')
    return parser

def get_frame(
        audio: np.ndarray,
        frame_start: int,
        frame_end: int,
        sample_rate: int = SAMPLE_RATE,
        return_timestamps: bool = False,        
):
    f"""
    Slice a frame from the audio tensor indicated by the sample indices `frame_start` and `frame_end`.
    If `return_timestamps=True`, instead return a dict with keys `start_s` (start time in seconds),
    `end_s` (end time in seconds) and `samples` (tensor of wav samples for the given frame).
    Pass `sample_rate` to override the default sample rate of {SAMPLE_RATE}.
    """
    samples = audio[frame_start:frame_end]
    if return_timestamps:
        frame_start_s = frame_start/sample_rate
        frame_end_s = frame_end/sample_rate
        return {
            'start_s': frame_start_s,
            'end_s': frame_end_s,
            'samples': samples
        }
    return samples

def get_sliding_window(
        audio: torch.Tensor,
        window_size: float,
        window_hop: float,
        sample_rate: int = SAMPLE_RATE,
        return_timestamps: bool = False,
    ):
    f"""
    Split audio tensor into a list of tensors, each corresponding to a frame of length `framelength_s`
    staggered by `frameshift_s`. If `return_timestamps=True`, return a list of dictionaries with keys `start_s`
    (start time in seconds), `end_s` (end time in seconds) and `samples` (tensor of wav samples for the given frame).
    Pass `sample_rate` to override the default sample rate of {SAMPLE_RATE}.
    """
    if len(audio)==0:
        return []
    framelength_samples = int(window_size * sample_rate)
    frameshift_samples = int(window_hop * sample_rate)
    
    frame_start = 0
    frame_end = framelength_samples
    windows = []
    while frame_end<len(audio):
        frame = get_frame(audio, frame_start, frame_end, sample_rate, return_timestamps)
        windows.append(frame)
        frame_start+=frameshift_samples
        frame_end+=frameshift_samples
    # append last truncated frame
    frame = get_frame(audio, frame_start, frame_end, sample_rate, return_timestamps)
    windows.append(frame)
    
    return windows

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
    seq_lens = torch.tensor(
        [seq.shape[0] for seq in batch_embeds],
        dtype=int,
        device=DEVICE,
    )
    padded_batch = pad_sequence(batch_embeds, batch_first=True, padding_value=0.0)
    padded_batch.to(DEVICE)
    return padded_batch, seq_lens

def prepare_embed_lists_for_decoding(
        query_embeds: List[torch.Tensor],
        test_embeds: List[torch.Tensor],
        distance_metric: Literal['cosine'] = 'cosine'
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Given a list of windowed embeddings for query and test phrases, pads
    lists and computes distance scores between embeddings, and returns
    lists of unpadded lengths (in number of windows) for each query and test phrase.
    Also handles batching test embeddings if num(test_embeds) > WFST_BATCH_SIZE.
    Assuming for now that query embeds will never exceed WFST_BATCH_SIZE,
    and so doesn't batch on query embeds.
    Note: the return values can be passed directly to `wfst.decode_keyword_batch`.

    Args:
        query_embeds: List of embeddings for query keyphrases
        test_embeds: List of embeddings for test phrases
        distance_metric: Type of distance metric to use.
            For now only 'cosine' is supported

    Returns: (distance_tensor, keyword_lens, seq_lens):
        torch.Tensor indicating distance scores,
        list of integers indicating unpadded query keyphrase lens
        and list of integers indicating unpadded test phrase lens
    """
    if distance_metric == 'cosine':
        distance_function = get_windowed_cosine_distance
    else:
        raise ValueError(f'Unknown distance metric: {distance_metric}')

    # get keyword lengths and pad
    keyword_lens = [query.shape[0] for query in query_embeds]
    query_embeds_padded = pad_sequence(query_embeds, batch_first=True, padding_value=0.0)
    query_embeds_padded.to(DEVICE)

    distance_tensor_list = []
    seq_lens = []
    dataloader = DataLoader(test_embeds, collate_fn=pad_and_return_lengths)
    for batch_test_embeds, batch_seq_lens in dataloader:
        batch_distance = distance_function(query_embeds_padded, batch_test_embeds)
        distance_tensor_list.extend(batch_distance)
        seq_lens.append(batch_seq_lens)

    distance_tensor = torch.concat(distance_tensor_list, dim=1)
    keyword_lens = torch.concat(seq_lens, dim=0)

    return distance_tensor, keyword_lens, seq_lens

"""
## CLAP IPA encoder utilities
- load_clap_speech_encoder: Load a CLAP speech encoder model.
- load_clap_speech_processor: Load the corresponding audio processor.
- encode_clap_audio: Encode a batch of audio samples into embeddings using the CLAP encoder and processor.
"""

def load_clap_speech_encoder(
    encoder_size: Literal['tiny', 'base', 'small'] = 'small'
) -> SpeechEncoder:
    """
    Load a CLAP speech encoder of the specified size.
    Available sizes are 'tiny', 'base', and 'small'.
    """
    encoder_name = CLAP_IPA_ENCODER_NAME.format(encoder_size=encoder_size)
    encoder = SpeechEncoder.from_pretrained(encoder_name)
    encoder.to(DEVICE)
    return encoder

def load_clap_speech_processor():
    """
    Load the CLAP speech processor (WhisperProcessor).
    The processor is the same for all CLAP speech encoder sizes.
    """
    return AutoProcessor.from_pretrained('openai/whisper-tiny')

def encode_clap_audio(
    audio_batch: Union[List[np.ndarray], torch.Tensor],
    speech_encoder: SpeechEncoder,
    speech_processor: AutoProcessor,
):
    """
    Encode a batch of audio samples into embeddings using the CLAP speech encoder and processor.
    Args:
        audio_batch (Union[List[np.ndarray], torch.Tensor]): Batch of audio samples. Each sample is a 1D numpy array or tensor.
        speech_encoder (SpeechEncoder): Pretrained CLAP speech encoder model.
        speech_processor (AutoProcessor): Corresponding audio processor (WhisperProcessor).
    Returns:
        torch.Tensor: Tensor of shape (batch_size, embedding_dim) containing the audio embeddings.
    """
    if type(audio_batch) is torch.Tensor:
        audio_batch = audio_batch.cpu().numpy()
    audio_input = speech_processor(
        audio_batch,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt',
        return_attention_mask=True,
    )
    audio_input=audio_input.to(DEVICE)

    with torch.no_grad():
        speech_embed = speech_encoder(**audio_input)['pooler_output']
    return speech_embed.cpu()

"""
## Speechbrain encoder utilities
"""

def load_speechbrain_encoder(
    model_name: str = SPEECHBRAIN_LID_ENCODER_NAME,
) -> EncoderClassifier:
    f"""
    Load a Speechbrain encoder model for speech embedding extraction.
    Default model is {SPEECHBRAIN_LID_ENCODER_NAME}.
    """
    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        run_opts={"device":DEVICE},
    )
    return classifier

def encode_speechbrain_audio(
    audio_batch: Union[List[np.ndarray], torch.Tensor],
    speechbrain_encoder: EncoderClassifier,
):
    """
    Encode a batch of audio samples into embeddings using the Speechbrain encoder.
    Args:
        audio_batch (Union[List[np.ndarray], torch.Tensor]): Batch of audio samples. Each sample is a 1D numpy array or tensor.
        speechbrain_encoder (EncoderClassifier): Pretrained Speechbrain encoder model.
    Returns:
        torch.Tensor: Tensor of shape (batch_size, embedding_dim) containing the audio embeddings.
    """
    if type(audio_batch) is np.ndarray:
        audio_batch = torch.tensor(audio_batch)
    elif type(audio_batch) is list:
        audio_batch = [torch.tensor(audio) for audio in audio_batch]
    audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=0.0)
    audio_batch = audio_batch.to(DEVICE)
    with torch.no_grad():
        speech_embed = speechbrain_encoder.encode_batch(audio_batch)
    audio_batch.cpu()
    speech_embed = speech_embed.squeeze(1).cpu()
    return speech_embed

"""
## Distance computation utilities
- compute_cosine_distance_matrix: Compute cosine distance matrix between two sets of embeddings.
"""


def get_cosine_distance(
        query_embeds: torch.Tensor,
        test_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine distance scores between query embeddings and test embeddings
    where the (i,j)^th element is the cosine distance between the i^th query
    embedding and the j^th test embedding.

    Args:
        query_embeds: Tensor of shape (num_queries, embed_dim)
        test_embeds: Tensor of shape (num_tests, embed_dim)
    Returns:
        Tensor of shape (num_queries, num_tests) with cosine distance scores
    """
    query_norm = query_embeds / query_embeds.norm(dim=1, keepdim=True)
    test_norm = test_embeds / test_embeds.norm(dim=1, keepdim=True)
    similarity_scores = torch.matmul(query_norm, test_norm.T)
    distance_scores = 1-similarity_scores
    return distance_scores


def get_windowed_cosine_distance(
        query_embeds: torch.Tensor,
        test_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Computes cosine distance scores between windowed query embeddings and test embeddings.
    Returns a 4-d tensor where each (i,j)^th element is a distance matrix between
    the windowed embeddings from the i^th query and j^th test phrase.

    Args:
        query_embeds: Tensor of shape (num_queries, num_windows, embed_dim)
        test_embeds: Tensor of shape (num_tests, num_windows, embed_dim)

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
    distance_scores = 1-similarity_scores
    return distance_scores
