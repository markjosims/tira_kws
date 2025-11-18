from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from constants import BATCH_SIZE, TIRA_ASR_PATH, TIRA_DRZ_PATH
from encoding import load_clap_speech_encoder, load_clap_speech_processor, encode_clap_audio
from typing import *

def load_tira_asr() -> Dataset:
    dataset = load_from_disk(TIRA_ASR_PATH)
    return dataset

def load_tira_drz() -> Dataset:
    dataset = load_from_disk(TIRA_DRZ_PATH)
    return dataset

def prepare_dataset(
        dataset,
        encoding: Literal['clap_ipa', None] = None,
        encoder_size: Literal['tiny', 'base', 'small'] = 'small',
    ):
    processor = load_clap_speech_processor()
    if encoding == 'clap_ipa':
        speech_encoder = load_clap_speech_encoder(encoder_size)
        speech_encoder.eval()
        encoder_funct = lambda audio, _: encode_clap_audio(audio, speech_encoder, processor)
    else:
        encoder_funct = lambda audio, sr: processor(audio, sampling_rate=sr, return_tensors='pt')['input_features']
    if isinstance(dataset, DatasetDict):
        colnames = dataset['train'].column_names
    else: # dataset is a Dataset
        colnames = dataset.column_names
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
    processed_batch['input_features'] = encoder_funct(audio, sr)


    transcription = batch['transcription']
    processed_batch['label_ids'] = processor.tokenizer(
        transcription,
        padding=True,
        return_tensors='pt',
    )['input_ids']

    return processed_batch

def get_audio_dataloader(
        dataset: Dataset,
        batch_size: int = BATCH_SIZE
    ) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)