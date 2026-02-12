from __future__ import annotations
from typing import *
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor
from argparse import ArgumentParser

from torchaudio.pipelines import (
    WAVLM_BASE, WAVLM_BASE_PLUS, WAVLM_LARGE,
    WAV2VEC2_XLSR_300M, WAV2VEC2_XLSR53, WAV2VEC2_XLSR_1B, WAV2VEC2_XLSR_2B,
)
from torchaudio.models.wav2vec2.model import Wav2Vec2Model
import librosa

from src.constants import (
    CLAP_IS_AVAILABLE, SPEECHBRAIN_IS_AVAILABLE, DEVICE, SAMPLE_RATE,
    SPEECHBRAIN_LID_ENCODER_NAME, CLAP_IPA_ENCODER_NAME, IPA_ALIGNER_ENCODER_NAME,
    WAV2VEC_DOWNSAMPLE_FACTOR
)

if CLAP_IS_AVAILABLE or TYPE_CHECKING:
    from clap.encoders import SpeechEncoder
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
        '--encoder',
        '-e',
        type=str,
        default='clap_ipa',
        choices=['clap_ipa', 'ipa_align', 'speechbrain_lid', 'wavlm', 'xlsr'],
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

def pool_embeds(
        embed_list: List[torch.Tensor],
        pooling_type: Literal['mean'] = 'mean',
) -> torch.Tensor:
    """
    Given a list of 2d tensors of shape (record_len, embed_dim),
    performs pooling on each record and returns as a 2d tensor
    of shape (num_records, embed_dim).

    Args:
        embed_list: List of 2d embedding tensors
        pooling_type: strategy for pooling each row of embeddings,
            for now only mean pooling is supported

    Returns:
        pooled_embeds: torch.Tensor of shape (num_records, embed_dim)
            containing pooled embedding tensors
    """
    if pooling_type == 'mean':
        pooling_funct = lambda t: torch.mean(t, dim=0)
    else:
        raise ValueError(f'Pooling type {pooling_type} is not supported')

    pooled_embeds = map(pooling_funct, embed_list)
    pooled_embeds = torch.stack(list(pooled_embeds))

    return pooled_embeds

"""
## CLAP IPA encoder utilities
- load_clap_speech_encoder: Load a CLAP speech encoder model.
- load_ipaalign_speech_encoder: Load an IPA-ALIGN speech encoder model.
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
    return encoder.eval().to(DEVICE)

def load_ipaalign_speech_encoder(
        encoder_size: Literal['tiny', 'base', 'small'] = 'small'
) -> SpeechEncoder:
    """
    Load an IPA-ALIGN speech encoder of the specified size.
    Available sizes are 'tiny', 'base', and 'small'.
    """
    encoder_name = IPA_ALIGNER_ENCODER_NAME.format(encoder_size=encoder_size)
    encoder = SpeechEncoder.from_pretrained(encoder_name)
    return encoder.eval().to(DEVICE)

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
- load_speechbrain_encoder: Load a Speech brain encoder model.
- encode_speechbrain_audio: Encode a batch of audio samples into embeddings
    using the Speechbrain encoder
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
## Wav2Vec family encoders
- load_xlsr: Load XLSR model
- load_wavlm: Load WavLM base-plus
- encode_wav2vec: Encode a batch of audio samples into embeddings
"""

def load_wavlm(
        encoder_size: Literal['base', 'base_plus', 'large']
) -> Wav2Vec2Model:
    if encoder_size=='base':
        encoder = WAVLM_BASE.get_model()
    elif encoder_size=='base_plus':
        encoder = WAVLM_BASE_PLUS.get_model()
    elif encoder_size=='large':
        encoder = WAVLM_LARGE.get_model()
    else:
        raise ValueError(f'Encoder size {encoder_size} is not supported.')
    return encoder.eval().to(DEVICE)

def load_xlsr(
        encoder_size: Literal['300m', '53', '1B', '2B']
) -> Wav2Vec2Model:
    if encoder_size=='300m':
        encoder = WAV2VEC2_XLSR_300M.get_model()
    elif encoder_size=='53':
        encoder = WAV2VEC2_XLSR53.get_model()
    elif encoder_size=='1B':
        encoder = WAV2VEC2_XLSR_1B.get_model()
    elif encoder_size=='2B':
        encoder = WAV2VEC2_XLSR_2B.get_model()
    else:
        raise ValueError(f'Encoder size {encoder_size} is not supported.')
    return encoder.eval().to(DEVICE)

def encode_wav2vec(
        audio_batch: List[np.ndarray],
        sample_rate: int,
        encoder: Wav2Vec2Model,
) -> Tuple[torch.Tensor, List[int]]:
    if sample_rate!=SAMPLE_RATE:
        audio_batch = [librosa.resample(
            y=audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        ) for audio in audio_batch]

    # cast audio arrays to tensor and pad
    audio_lens = [len(audio)//WAV2VEC_DOWNSAMPLE_FACTOR for audio in audio_batch]
    audio_lens = torch.tensor(audio_lens)
    audio_batch = [torch.tensor(audio) for audio in audio_batch]
    audio_batch = pad_sequence(audio_batch, batch_first=True)
    audio_batch = audio_batch.to(DEVICE)

    with torch.no_grad():
        features, _ = encoder.extract_features(
            audio_batch,
        )
    last_layer = features[-1]
    last_layer = last_layer.cpu()


    # `prepare_dataset_batch` expects a 2d tensor
    # of concatenated embeddings w/ padding removed
    embeds = []
    for i, audio_len in enumerate(audio_lens):
        embeds.append(last_layer[i, :audio_len])
    embeds_flat = torch.cat(embeds, dim=0)

    # update audio_lens in case of off-by-one errors
    audio_lens = torch.tensor([embed.shape[0] for embed in embeds])

    return embeds_flat, audio_lens
