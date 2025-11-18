"""
Script to compute cosine similarity matrix for audio records from a dataset.
Currently only supports within-dataset similarity for Tira ASR dataset.
TODO: Extend to cross-dataset similarity between Tira ASR and Tira DRZ datasets.
"""

from constants import DEVICE
from dataloading import load_tira_asr, get_audio_dataloader, prepare_dataset
from encoding import (
    encode_clap_audio, compute_cosine_similarity_matrix,
    load_clap_speech_encoder, load_clap_speech_processor,
)
import torch
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Compute dataset similarity matrix")
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
    dataset = load_tira_asr()
    dataset = dataset['train']
    dataset = prepare_dataset(
        dataset,
        encoding='clap_ipa',
        encoder_size=args.encoder_size,
    )
    if args.num_records is not None:
        dataset = dataset.select(range(args.num_records))

    speech_encoder = load_clap_speech_encoder(args.encoder_size)
    speech_encoder.eval()


    outer_dataloader = get_audio_dataloader(dataset, batch_size=args.batch_size)
    inner_dataloader = get_audio_dataloader(dataset, batch_size=args.batch_size)
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
    
    torch.save(similarity_matrix, args.output)
    

if __name__ == "__main__":
    main()