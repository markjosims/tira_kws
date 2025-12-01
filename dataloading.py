from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader
import torch
from transformers import WhisperProcessor
from constants import BATCH_SIZE, TIRA_ASR_PATH, TIRA_DRZ_PATH
from encoding import (
    load_clap_speech_encoder, load_clap_speech_processor, encode_clap_audio,
    get_sliding_window,
)
from typing import *
from argparse import ArgumentParser

def add_dataset_optional_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        '--num_records', '-n', type=int, default=None, help='Number of records to load from dataset.'
    )
    parser.add_argument(
        '--batch_size', '-b', type=int, default=BATCH_SIZE, help='Batch size for data loading and computing embeddings.'
    )
    return parser

def add_dataset_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        '--dataset', '-d', type=str, default='tira_asr', help='Dataset to load.'
    )
    add_dataset_optional_args(parser)
    return parser

def load_tira_asr() -> Dataset:
    dataset = load_from_disk(TIRA_ASR_PATH)

    # combine train, validation, and test splits
    dataset = concatenate_datasets([
        dataset['train'],
        dataset['validation'],
        dataset['test'],
    ])

    return dataset

def load_tira_drz() -> Dataset:
    dataset = load_from_disk(TIRA_DRZ_PATH)
    dataset = dataset.rename_columns({'text': 'transcription'})
    dataset = dataset['train']

    # for now only returning English subset
    dataset = dataset.filter(lambda example: example['transcription'].lower() == 'eng')
    return dataset

def load_dataset(dataset_name: str, num_records: Optional[int] = None) -> Dataset:
    if dataset_name == "tira_asr":
        dataset = load_tira_asr()
    elif dataset_name == "tira_drz":
        dataset = load_tira_drz()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if num_records is not None:
        dataset = dataset.select(range(num_records))
    return dataset
    

def get_encoder_funct_w_sliding_window(
        encoder_funct: Callable,
        window_size: float,
        window_hop: Optional[float] = None,
        batch_size: int = BATCH_SIZE,
    ) -> Callable:
    if window_hop is None:
        window_hop = window_size / 2.0
    original_encoder_funct = encoder_funct
    def encoder_funct(audio_batch, sr):
        audio_batch = [torch.tensor(audio) for audio in audio_batch]
        window_encodings = []
        num_windows_per_record = []
        windows = []
        for audio in audio_batch:
            window_batch = get_sliding_window(
                audio,
                sample_rate=sr,
                window_size=window_size,
                window_hop=window_hop,
            )

            windows.extend(window_batch)
            num_windows_per_record.append(len(window_batch))
        windows = pad_sequence(windows, batch_first=True, padding_value=0.0)
        window_batchloader = DataLoader(windows, batch_size=batch_size)
        for batch in tqdm(window_batchloader, total=len(window_batchloader)):
            window_embeddings, _ = original_encoder_funct(batch, sr)
            window_encodings.append(window_embeddings)
        all_window_embeddings = torch.cat(window_encodings, dim=0)
        return all_window_embeddings, num_windows_per_record
    return encoder_funct

def prepare_dataset(
        dataset,
        encoder: Literal['clap_ipa', None] = None,
        encoder_size: Literal['tiny', 'base', 'small'] = 'small',
        window_size: Optional[float] = None,
        window_hop: Optional[float] = None,
        batch_size: int = BATCH_SIZE,
    ) -> Dataset:
    processor = load_clap_speech_processor()
    if encoder == 'clap_ipa':
        speech_encoder = load_clap_speech_encoder(encoder_size)
        speech_encoder.eval()
        encoder_funct = lambda audio, _: (encode_clap_audio(
            audio, speech_encoder, processor
        ), None)
    elif encoder == 'speechbrain_lid':
        speechbrain_encoder = load_speechbrain_encoder()
        speechbrain_encoder.eval()
        encoder_funct = lambda audio, _: (encode_speechbrain_audio(
            audio, speechbrain_encoder,
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
            batch_size,
        )

    dataset = dataset.map(
        prepare_dataset_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=colnames,
        load_from_cache_file=True,
        with_indices=True,
        fn_kwargs={
            'processor': processor,
            'encoder_funct': encoder_funct,
        }
    )
    dataset.set_format('torch')
    
    return dataset

def prepare_dataset_batch(
        batch: Dataset,
        index: int,
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
        label_ids = expand_col(num_windows, label_ids)
        index = expand_col(num_windows, index)
    processed_batch['label_ids'] = label_ids
    processed_batch['index'] = index

    return processed_batch

def expand_col(num_windows, col):
    expanded_col = []
    for i, n_windows in enumerate(num_windows):
        for _ in range(n_windows):
            expanded_col.append(col[i])
    if type(col) == torch.Tensor:
        col = torch.stack(expanded_col, dim=0)
    else:
        col = expanded_col
    return col

def get_audio_dataloader(
        dataset: Dataset,
        batch_size: int = BATCH_SIZE,
    ) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)