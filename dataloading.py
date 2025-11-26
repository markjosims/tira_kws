from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
import torch
from transformers import WhisperProcessor
from constants import BATCH_SIZE, TIRA_ASR_PATH, TIRA_DRZ_PATH
from encoding import (
    load_clap_speech_encoder, load_clap_speech_processor, encode_clap_audio,
    get_sliding_window,
)
from typing import *

def load_tira_asr() -> Dataset:
    dataset = load_from_disk(TIRA_ASR_PATH)
    return dataset

def load_tira_drz() -> Dataset:
    dataset = load_from_disk(TIRA_DRZ_PATH)
    dataset = dataset.rename_columns({'text': 'transcription'})

    # for now only returning English subset
    dataset = dataset.filter(lambda example: example['transcription'].lower() == 'eng')
    return dataset

def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name == "tira_asr":
        return load_tira_asr()
    elif dataset_name == "tira_drz":
        return load_tira_drz()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_encoder_funct_w_sliding_window(
        encoder_funct: Callable,
        window_size: float,
        window_hop: Optional[float] = None,
    ) -> Callable:
    if window_hop is None:
        window_hop = window_size / 2.0
    original_encoder_funct = encoder_funct
    def encoder_funct(audio_batch, sr):
        window_encodings = []
        num_windows = []
        for audio in audio_batch:
            windows = get_sliding_window(
                audio,
                sample_rate=sr,
                window_size=window_size,
                window_hop=window_hop,
            )
            window_embeddings, _ = original_encoder_funct(windows, sr)
            window_encodings.append(window_embeddings)
            num_windows.append(len(windows))
        all_window_embeddings = torch.cat(window_encodings, dim=0)
        return all_window_embeddings, num_windows
    return encoder_funct

def prepare_dataset(
        dataset,
        encoder: Literal['clap_ipa', None] = None,
        encoder_size: Literal['tiny', 'base', 'small'] = 'small',
        window_size: Optional[float] = None,
        window_hop: Optional[float] = None,
    ) -> Dataset:
    processor = load_clap_speech_processor()
    if encoder == 'clap_ipa':
        speech_encoder = load_clap_speech_encoder(encoder_size)
        speech_encoder.eval()
        encoder_funct = lambda audio, _: (encode_clap_audio(
            audio, speech_encoder, processor
        ), None)
    else:
        encoder_funct = lambda audio, sr: (processor(
            audio, sampling_rate=sr, return_tensors='pt'
        )['input_features'], None)

    if isinstance(dataset, DatasetDict):
        colnames = dataset['train'].column_names
    else: # dataset is a Dataset
        colnames = dataset.column_names

    if window_size is not None:
        encoder_funct = get_encoder_funct_w_sliding_window(
            encoder_funct,
            window_size,
            window_hop,
        )

    dataset = dataset.map(
        prepare_dataset_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=colnames,
        load_from_cache_file=True,
        fn_kwargs={
            'processor': processor,
            'encoder_funct': encoder_funct,
        }
    )
    dataset.set_format('torch')
    
    return dataset

def prepare_dataset_batch(
        batch: Dataset,
        processor: WhisperProcessor,
        encoder_funct: Callable = None,
) -> Dataset:
    processed_batch = {}

    audio = [row['array'] for row in batch['audio']]
    sr = batch['audio'][0]['sampling_rate']
    input_features, num_windows = encoder_funct(audio, sr)
    processed_batch['input_features'] = input_features


    transcription = batch['transcription']
    label_ids = processor.tokenizer(
        transcription,
        padding=True,
        return_tensors='pt',
    )['input_ids']

    if num_windows is not None:
        label_ids = expand_label_ids(num_windows, label_ids)
    processed_batch['label_ids'] = label_ids

    return processed_batch

def expand_label_ids(num_windows, label_ids):
    expanded_label_ids = []
    for i, n_windows in enumerate(num_windows):
        for _ in range(n_windows):
            expanded_label_ids.append(label_ids[i])
    label_ids = torch.stack(expanded_label_ids, dim=0)
    return label_ids

def get_audio_dataloader(
        dataset: Dataset,
        batch_size: int = BATCH_SIZE
    ) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)