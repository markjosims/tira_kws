from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from constants import BATCH_SIZE, TIRA_ASR_PATH, TIRA_DRZ_PATH
from encoding import load_clap_speech_processor

def load_tira_asr() -> Dataset:
    dataset = load_from_disk(TIRA_ASR_PATH)
    dataset = prepare_dataset(dataset)
    return dataset

def load_tira_drz() -> Dataset:
    dataset = load_from_disk(TIRA_DRZ_PATH)
    dataset = prepare_dataset(dataset)
    return dataset

def prepare_dataset(dataset):
    processor = load_clap_speech_processor()
    dataset = dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=dataset['train'].column_names,
    )
    dataset.set_format('torch')
    
    return dataset

def prepare_dataset_batch(
        batch: Dataset,
        processor: WhisperProcessor,
) -> Dataset:
    processed_batch = {}

    audio = [row['array'] for row in batch['audio']]
    processed_batch['input_features'] = processor(
        audio,
        sampling_rate=batch['audio'][0]['sampling_rate'],
        return_tensors='pt'
    )['input_features']


    transcription = batch['transcription']
    processed_batch['label_ids'] = processor.tokenizer(
        transcription,
        padding=True,
        return_tensors='pt',
    )['input_ids']

    return processed_batch

def get_audio_dataloader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE)
