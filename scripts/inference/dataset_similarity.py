"""
Script to compute cosine similarity matrix for audio records from a dataset.
Currently only supports within-dataset similarity for Tira ASR dataset.
TODO: Extend to cross-dataset similarity between Tira ASR and Tira DRZ datasets.
"""

from src.constants import DEVICE
from src.dataloading import load_tira_asr, load_tira_drz, get_audio_dataloader, prepare_dataset
from distance import (
    encode_clap_audio, compute_cosine_similarity_matrix,
    load_clap_speech_encoder, load_clap_speech_processor,
)
import torch
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Compute dataset similarity matrix")
    parser.add_argument(
        '--dataset', '-d', type=str, default='tira_asr', help='Dataset to compute similarity for.'
    )
    parser.add_argument(
        '--drz_dataset', '--drz', type=str, default=None, help='Inner dataset for cross-dataset similarity.'
    )
    parser.add_argument(
        '--drz_dataset_window_size', '-w', type=float, default=None, help='Window size for inner dataset sliding window.'
    )
    parser.add_argument(
        "--batch-size", '-b', type=int, default=128, help="Batch size for audio embeddings."
    )
    parser.add_argument(
        '--output', '-o', type=str, default='data/similarity_matrix.pt', help='Output file for similarity matrix'
    )
    parser.add_argument(
        '--num_records', '-n', type=int, default=None, help='Number of records to process from the dataset (useful for testing).'
    )
    parser.add_argument('--encoder_size', '-e', type=str, default='base', help='Size of CLAP speech encoder to use.')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = load_main_dataset(args)
    outer_dataloader = get_audio_dataloader(dataset, batch_size=args.batch_size)

    inner_dataset = load_inner_dataset(args, dataset)
    inner_dataloader = get_audio_dataloader(inner_dataset, batch_size=args.batch_size)
    similarity_matrix = torch.zeros(
        len(outer_dataloader.dataset), len(inner_dataloader.dataset)
    ).to(DEVICE)

    for i, batch in tqdm(
        enumerate(outer_dataloader),
        total=len(outer_dataloader)
    ):
        outer_start = i * outer_dataloader.batch_size
        outer_end = outer_start + len(batch['input_features'])
        outer_embeddings = batch['input_features']
        for j, inner_batch in enumerate(inner_dataloader):
            inner_start = j * inner_dataloader.batch_size
            inner_end = inner_start + len(inner_batch['input_features'])
            inner_embeddings = inner_batch['input_features']
            batch_matrix = compute_cosine_similarity_matrix(outer_embeddings, inner_embeddings)
            similarity_matrix[
                outer_start:outer_end,
                inner_start:inner_end
            ] = batch_matrix
    
    output_path = get_similarity_matrix_path(args)
    torch.save(similarity_matrix, output_path)

def get_similarity_matrix_path(args):
    if args.drz_dataset is None:
        return args.output
    else:
        return args.output.replace(
            '.pt',
            f'_{args.drz_dataset}'
            + (f'_{args.drz_dataset_language}' if args.drz_dataset_language else '')
            + (f'_ws{args.drz_dataset_window_size}'.replace('.', '_') if args.drz_dataset_window_size else '')
            + '.pt'
        )

def load_inner_dataset(args, dataset):
    if (args.drz_dataset is None) or (args.drz_dataset == args.dataset):
        inner_dataset = dataset
    elif args.drz_dataset == 'tira_drz':
        inner_dataset = load_tira_drz()
        inner_dataset = inner_dataset['train']
    else:
        raise ValueError(f"Unsupported inner dataset: {args.drz_dataset}")

    if args.num_records is not None:
        inner_dataset = inner_dataset.select(range(args.num_records))

    inner_dataset = prepare_dataset(
        inner_dataset,
        encoder='clap_ipa',
        encoder_size=args.encoder_size,
        window_size=args.drz_dataset_window_size,
    )
    return inner_dataset

def load_main_dataset(args):
    if args.dataset == 'tira_asr':
        dataset = load_tira_asr()
    else:
        raise ValueError(f"Unsupported inner dataset: {args.dataset}")
    
    dataset = dataset['train']
    if args.num_records is not None:
        dataset = dataset.select(range(args.num_records))
    dataset = prepare_dataset(
        dataset,
        encoder='clap_ipa',
        encoder_size=args.encoder_size,
    )
    return dataset
    

if __name__ == "__main__":
    main()