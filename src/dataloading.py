import datasets
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import WhisperProcessor
from src.constants import BATCH_SIZE, TIRA_ASR_PATH, TIRA_ASR_URI, TIRA_DRZ_PATH
from src.encoding import (
    load_clap_speech_encoder, load_clap_speech_processor, encode_clap_audio,
    load_speechbrain_encoder, encode_speechbrain_audio,
    get_sliding_window, load_ipaalign_speech_encoder,
    load_xlsr, load_wavlm, encode_wav2vec
)
from typing import *
from argparse import ArgumentParser
import os

def add_dataset_optional_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Add optional parameters for dataset loading to the argument parser.
    Keep separate from `add_dataset_args` to allow reuse in other scripts
    where the `--dataset` arg isn't used, e.g. `lid_eval.py` which
    uses `--in_domain` and `--out_domain` instead.
    """
    parser.add_argument(
        '--num_records', '-n', type=int, default=None, help='Number of records to load from dataset.'
    )
    parser.add_argument(
        '--batch_size', '-b', type=int, default=BATCH_SIZE, help='Batch size for data loading and computing embeddings.'
    )
    return parser

def add_dataset_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Add dataset-related parameters to the argument parser.
    """
    parser.add_argument(
        '--dataset', '-d', type=str, default='tira_asr', help='Dataset to load.'
    )
    add_dataset_optional_args(parser)
    return parser

def load_tira_asr() -> datasets.Dataset:
    """
    Load the Tira ASR dataset from disk and combine train, validation, and test splits
    (since the KWS experiment doesn't involve training).
    """
    if os.path.exists(TIRA_ASR_PATH):
        dataset = datasets.load_from_disk(TIRA_ASR_PATH)
    else:
        dataset = datasets.load_dataset(TIRA_ASR_URI)

    # combine train, validation, and test splits
    dataset = datasets.concatenate_datasets([
        dataset['train'],
        dataset['validation'],
        dataset['test'],
    ])

    return dataset

def load_tira_drz() -> datasets.Dataset:
    """
    Load the Tira Diarization dataset from disk, filter non-English rows
    and return the 'train' (only) subset. 
    """
    dataset = datasets.load_from_disk(TIRA_DRZ_PATH)
    dataset = dataset.rename_columns({'text': 'transcription'})
    dataset = dataset['train']

    # for now only returning English subset
    dataset = dataset.filter(lambda example: example['transcription'].lower() == 'eng')
    return dataset

def load_dataset(dataset_name: str, num_records: Optional[int] = None) -> datasets.Dataset:
    """
    Load dataset by name.
    - `tira_asr`: Tira ASR dataset
    - `tira_drz`: Tira Diarization dataset
    """
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
    """
    Wrap a function that encodes audio batches to apply sliding window encoding.
    Normal behavior is for the encoder function to take in a batch of audio samples.
    Since windowing breaks each audio sample into multiple windows, the new encoder function
    will first break each audio sample into windows, then encode all windows in batches,
    and finally return all window encodings concatenated together.

    In order to keep track of which windows belong to which original audio sample,
    the new encoder function will also return a list indicating the number of windows
    generated for each original audio sample.

    Arguments:
        - encoder_funct: Original encoder function that takes in audio batch and sample rate.
        - window_size: Size of the sliding window in seconds.
        - window_hop: Hop size between windows in seconds. If None, defaults to half the window size.
        - batch_size: Batch size for processing windows.
    Returns:
        - A new encoder function that applies sliding window encoding and returns the concatenated window
            embeddings along with the number of windows per original audio sample.
    """
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
        with tqdm(total=len(window_batchloader)*batch_size) as pbar:
            for batch in window_batchloader:
                window_embeddings, _ = original_encoder_funct(batch, sr)
                window_encodings.append(window_embeddings)
                pbar.update(batch_size)
        all_window_embeddings = torch.cat(window_encodings, dim=0)
        return all_window_embeddings, num_windows_per_record
    return encoder_funct

def prepare_dataset(
        dataset,
        encoder: Literal[
            'clap_ipa', 'speechbrain', 'wavlm', 'xlsr',
        ] = None,
        encoder_size: Literal['tiny', 'base', 'small'] = 'small',
        window_size: Optional[float] = None,
        window_hop: Optional[float] = None,
        batch_size: int = BATCH_SIZE,
    ) -> datasets.Dataset:
    """
    Applies audio preprocessing and, optionally, encoding to the dataset.
    If no `encoder` is specified, audio is preprocessed using the WhisperProcessor only.
    If `window_size` is specified, sliding window encoding is applied.
    Returns the processed dataset with columns `input_features` and `label_ids`.
    If sliding window encoding is used, the dataset will be enlarged so that `input_features`
    will contain embeddings for each window, and `label_ids` will be expanded accordingly to
    match the number of windows, and an additional `index` column will be added to keep track
    of the original record indices.
    """
    processor = load_clap_speech_processor()
    if encoder == 'clap_ipa':
        speech_encoder = load_clap_speech_encoder(encoder_size)
        speech_encoder.eval()
        encoder_funct = lambda audio, _: (encode_clap_audio(
            audio, speech_encoder, processor
        ), None)
    elif encoder == 'ipa_align':
        speech_encoder = load_ipaalign_speech_encoder(encoder_size)
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
    elif encoder == 'wavlm':
        encoder = load_wavlm(encoder_size)
        encoder_funct = lambda audio, sr: encode_wav2vec(audio, sr, encoder)
    elif encoder == 'xlsr':
        encoder = load_xlsr(encoder_size)
        encoder_funct = lambda audio, sr: encode_wav2vec(audio, sr, encoder)

    else:
        def encoder_funct(audio, sr):
            if type(audio) is torch.Tensor:
                audio = audio.numpy()
            input_features = processor(
                audio, sampling_rate=sr, return_tensors='pt'
            )['input_features']
            return input_features, None

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
        batch: datasets.Dataset,
        index: int,
        processor: WhisperProcessor,
        encoder_funct: Callable,
) -> datasets.Dataset:
    """
    Helper function for `prepare_dataset`. Prepare a batch of dataset by encoding audio and
    processing labels. `encoder_funct` is a function that takes in a batch of audio samples
    and sample rate and returns a tuple of encoded input features and an array of window counts
    (if using sliding window encoding) or None.

    If using sliding window encoding, use the `num_windows` array to expand the `label_ids` and `index`
    columns to match the number of windows.
    """
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
    """
    Arguments:
    - num_windows: List of number of windows per original audio sample.
    - col: Column to expand (list or tensor).
    Returns:
    - Expanded column (list or tensor) where each entry in `col` is repeated
      according to the corresponding number in `num_windows`.

    Given a list `num_windows` indicating how many windows were generated
    for each original audio sample, expand the input column `col` by repeating
    each entry according to the corresponding number in `num_windows`.
    """
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
        dataset: datasets.Dataset,
        batch_size: int = BATCH_SIZE,
    ) -> DataLoader:
    """
    Wraps a dataset in a DataLoader with the specified batch size.
    """
    return DataLoader(dataset, batch_size=batch_size)