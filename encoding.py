from typing import *
from clap.encoders import SpeechEncoder, PhoneEncoder
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoProcessor
from constants import DEVICE, SAMPLE_RATE

#####################
# speech embeddings #
#####################

def load_clap_speech_encoder(
    encoder_size: Literal['tiny', 'base', 'small'] = 'small'
) -> SpeechEncoder:
    encoder_name = f'anyspeech/clap-ipa-{encoder_size}-speech'
    encoder = SpeechEncoder.from_pretrained(encoder_name)
    encoder.to(DEVICE)
    return encoder

def load_clap_speech_processor():
    return AutoProcessor.from_pretrained('openai/whisper-tiny')

def encode_clap_audio(
    audio_batch,
    speech_encoder,
    speech_processor,
):
    audio_input = speech_processor(
        audio_batch,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt',
        return_attention_mask=True,
    )
    audio_input=audio_input.to(DEVICE)

    with torch.no_grad():
        speech_embed = speech_encoder(**audio_input)['pooler_output']
    return speech_embed

def get_frame(
        audio: torch.Tensor,
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
    if return_timestamps:
        frame_start_s = frame_start/sample_rate
        frame_end_s = frame_end/sample_rate
        return {
            'start_s': frame_start_s,
            'end_s': frame_end_s,
            'samples': audio[frame_start:frame_end]
        }
    return audio[frame_start:frame_end]

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
    frame = get_frame(audio, frame_start, len(audio), sample_rate, return_timestamps)
    windows.append(frame)
    
    return windows

########################
# embedding comparison #
########################

def compute_cosine_similarity_matrix(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between two sets of embeddings.

    Args:
        embeddings_a (torch.Tensor): Tensor of shape (N, D)
        embeddings_b (torch.Tensor): Tensor of shape (M, D)

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (N, M)
    """

    return F.cosine_similarity(embeddings_a.unsqueeze(1), embeddings_b.unsqueeze(0), dim=-1)