"""
Script to cache speech embeddings for a dataset. Saves embeddings, mean embedding, and std embedding to disk.
Also saves whitened embeddings (z-score normalized) to disk. If a z-score dataset is provided, also saves embeddings
whitened using the mean and std from that dataset.

Arguments:
    --dataset: Name of the dataset to process.
    --encoder: Encoder model to use for generating embeddings.
    --encoder_size: Size of the encoder model.
    --zscore_dataset: (Optional) Name of dataset to use for z-score normalization.
"""

from argparse import ArgumentParser
from constants import EMBEDDINGS_DIR
from dataloading import load_dataset, prepare_dataset
import torch
from sklearn.manifold import TSNE

def parse_args():
    parser = ArgumentParser(description="Compute dataset similarity matrix")
    parser.add_argument(
        '--dataset', '-d', type=str, default='tira_asr', help='Dataset to compute similarity for.'
    )
    parser.add_argument('--encoder', '-m', type=str, default='clap_ipa', help='Encoder model to use for generating embeddings.')
    parser.add_argument('--encoder_size', '-s', type=str, default='base', help='Size of CLAP speech encoder to use.')
    parser.add_argument('--zscore_dataset', '-z', type=str, default=None, help='Name of dataset to use for z-score normalization.')
    return parser.parse_args()

def cache_embeddings(dataset: str, encoder: str, encoder_size: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Caches embeddings for the specified dataset using the given encoder.

    Args:
        dataset (str): The name of the dataset to process.
        encoder (str): The encoder model to use for generating embeddings.
        encoder_size (str): The size variant of the encoder model.
    """
    # Load the dataset
    print(f"Loading dataset {dataset}...")
    ds = load_dataset(dataset)
    if 'train' in ds:
        ds = ds['train']

    # Prepare the dataset for embedding extraction
    ds_encoded = prepare_dataset(ds, encoder, encoder_size)
    embeds = ds_encoded['input_features'][:]
    mean_embed = embeds.mean(dim=0)
    std_embed = embeds.std(dim=0)


    
    return embeds, mean_embed, std_embed

def whiten_embeddings(embeds: torch.Tensor, mean_embed: torch.Tensor, std_embed: torch.Tensor) -> torch.Tensor:
    return (embeds - mean_embed) / std_embed

def main():
    args = parse_args()
    embeds, mean_embed, std_embed = cache_embeddings(args.dataset, args.encoder, args.encoder_size)

    embeds_path = str(EMBEDDINGS_DIR / f"{args.dataset}_{args.encoder}_{args.encoder_size}_embeddings.pt")
    mean_embed_path = embeds_path.replace("_embeddings.pt", "_mean_embedding.pt") 
    std_embed_path = embeds_path.replace("_embeddings.pt", "_std_embedding.pt")

    print(f"Saving embeddings to {embeds_path}...")
    torch.save(embeds, embeds_path)
    print(f"Saving mean embedding to {mean_embed_path}...")
    torch.save(mean_embed, mean_embed_path)
    print(f"Saving std embedding to {std_embed_path}...")
    torch.save(std_embed, std_embed_path)

    embeds_whitened = whiten_embeddings(embeds, mean_embed, std_embed)
    embeds_whitened_path = embeds_path.replace("_embeddings.pt", "_whitened_embeddings.pt")
    print(f"Saving whitened embeddings to {embeds_whitened_path}...")
    torch.save(embeds_whitened, embeds_whitened_path)

    if args.zscore_dataset is not None:
        zscore_mean_path = mean_embed_path.replace(args.dataset, args.zscore_dataset)
        zscore_std_path = std_embed_path.replace(args.dataset, args.zscore_dataset)

        zscore_mean_embed = torch.load(zscore_mean_path)
        zscore_std_embed = torch.load(zscore_std_path)

        embeds_cross_whitened = whiten_embeddings(embeds, zscore_mean_embed, zscore_std_embed)
        embeds_cross_whitened_path = embeds_path.replace("_embeddings.pt", f"_whitened_from_{args.zscore_dataset}_embeddings.pt")

        print(f"Saving cross-whitened embeddings to {embeds_cross_whitened_path}...")
        torch.save(embeds_cross_whitened, embeds_cross_whitened_path)

if __name__ == "__main__":
    main()