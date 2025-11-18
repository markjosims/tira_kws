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