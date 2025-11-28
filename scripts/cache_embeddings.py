"""
Script to cache speech embeddings for a dataset. Saves embeddings, mean embedding, and std embedding to disk.
Also saves whitened embeddings (z-score normalized) to disk. If a z-score dataset is provided, also saves embeddings
whitened using the mean and std from that dataset.

Arguments:
    --dataset: Name of the dataset to process.
    --encoder: Encoder model to use for generating embeddings.
    --encoder_size: Size of the encoder model.
    --whitened_from_dataset: (Optional) Name of dataset to use for z-score normalization.
"""

from argparse import ArgumentParser
import os
from constants import EMBEDDINGS_DIR
from dataloading import load_dataset, prepare_dataset, add_dataset_args
from encoding import add_sliding_window_args, add_encoder_args
import torch
from typing import *

def parse_args():
    parser = ArgumentParser(description="Compute dataset similarity matrix")
    parser = add_dataset_args(parser)
    parser.add_argument('--whitened_from_dataset', '-z', type=str, default=None, help='Name of dataset to use for z-score normalization.')
    parser = add_sliding_window_args(parser)
    parser = add_encoder_args(parser)
    return parser.parse_args()

def get_embed_path(
        dataset: str,
        encoder: str,
        encoder_size: str,
        window_size: Optional[float] = None,
        embedding_type: Literal['regular', 'mean', 'std', 'whitened', 'indices'] = 'regular',
        whitened_from_dataset: Optional[str] = None,
        cross_whiten: bool = False,
        num_records: Optional[int] = None,
        **_, # to allow for unused args
) -> str:
    
    if whitened_from_dataset is not None and cross_whiten:
        embed_type_str = f"whitened_from_{whitened_from_dataset}_embeddings"
    elif embedding_type == 'indices':
        embed_type_str = "indices"
    elif embedding_type == 'whitened':
        embed_type_str = "whitened_embeddings"
    elif embedding_type != 'regular':
        embed_type_str = f"{embedding_type}_embedding"
    else:
        embed_type_str = "embeddings"

    embed_path_partial = EMBEDDINGS_DIR / f"{dataset}_{encoder}_{encoder_size}"
    embed_path_partial = str(embed_path_partial)
    
    if window_size is not None:
        window_size_str = str(window_size).replace('.', '_')
        embed_path_partial += f"_ws{window_size_str}"
    if num_records is not None:
        embed_path_partial += f"_nr{num_records}"

    embed_path = embed_path_partial + f"_{embed_type_str}.pt"
        
    return str(embed_path)

def get_all_embed_paths(argdict) -> Dict[str, str]:
    """
    Returns a dictionary of all relevant embedding paths for the given argument dictionary.
    """
    paths = {}
    paths['embeddings'] = get_embed_path(**argdict, embedding_type='regular')
    paths['mean'] = get_embed_path(**argdict, embedding_type='mean')
    paths['std'] = get_embed_path(**argdict, embedding_type='std')
    paths['whitened'] = get_embed_path(**argdict, embedding_type='whitened')
    whitened_from_dataset = argdict.get('whitened_from_dataset', None)
    if whitened_from_dataset is not None:
        paths['cross_whitened'] = get_embed_path(
            **argdict,
            embedding_type='whitened',
            cross_whiten=True,
        )
    if argdict.get('window_size', None) is not None:
        paths['indices'] = get_embed_path(**argdict, embedding_type='indices')
    return paths

def load_embeddings(argdict) -> Dict[str, torch.Tensor]:
    """
    Loads all relevant embeddings for the given argument dictionary.
    If any embeddings are missing, computes and caches them first.

    Args:
        argdict (Dict[str, Any]): Dictionary of arguments including dataset, encoder, etc.  
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing loaded embeddings.
    """
    embed_dict = {}
    paths = get_all_embed_paths(argdict)
    if not all(os.path.exists(path) for path in paths.values()):
        print("Embedding file not found, computing embeddings and caching...")
        breakpoint()
        return cache_embeddings(argdict)

    for key, path in paths.items():
        print(f"Loading {key} from {path}...")
        embed_dict[key] = torch.load(path)
    return embed_dict
    

def compute_embeddings(
        dataset: str = None,
        num_records: Optional[int] = None,
        encoder: str = None,
        encoder_size: str = None,
        window_size: Optional[float] = None,
        window_hop: Optional[float] = None,
        **_, # to allow for unused args
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Caches embeddings for the specified dataset using the given encoder.

    Args:
        dataset (str): The name of the dataset to process.
        encoder (str): The encoder model to use for generating embeddings.
        encoder_size (str): The size variant of the encoder model.
    """
    # Load the dataset
    print(f"Loading dataset {dataset}...")
    ds = load_dataset(dataset, num_records=num_records)
    if 'train' in ds:
        ds = ds['train']

    # Prepare the dataset for embedding extraction
    ds_encoded = prepare_dataset(
        dataset=ds,
        encoder=encoder,
        encoder_size=encoder_size,
        window_size=window_size,
        window_hop=window_hop,
    )
    embeds = ds_encoded['input_features'][:]
    indices = None
    if window_size is not None:
        indices = ds_encoded['index'][:]
    mean_embed = embeds.mean(dim=0)
    std_embed = embeds.std(dim=0)

    
    return embeds, mean_embed, std_embed, indices

def whiten_embeddings(embeds: torch.Tensor, mean_embed: torch.Tensor, std_embed: torch.Tensor) -> torch.Tensor:
    """
    Z-score normalizes the embeddings using the provided mean and std embeddings.
    """
    return (embeds - mean_embed) / std_embed


def cache_embeddings(argdict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Computes and caches embeddings for a dataset and returns them in a dictionary.
    Also computes and caches mean and std embeddings, as well as whitened embeddings
    which are z-score normalized. If a z-score dataset is provided in argdict,
    also computes and caches embeddings whitened using the mean and std from that dataset.

    Args:
        argdict (Dict[str, Any]): Dictionary of arguments including dataset, encoder, etc.
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing embeddings, mean, std, and whitened embeddings
    """
    embed_dict = {}

    embeds, mean_embed, std_embed, indices = compute_embeddings(**argdict)
    embeds_path = get_embed_path(**argdict)
    mean_embed_path = get_embed_path(
        embedding_type='mean',
        **argdict,
    )
    std_embed_path = get_embed_path(
        embedding_type='std',
        **argdict,
    )

    print(f"Saving embeddings to {embeds_path}...")
    torch.save(embeds, embeds_path)
    print(f"Saving mean embedding to {mean_embed_path}...")
    torch.save(mean_embed, mean_embed_path)
    print(f"Saving std embedding to {std_embed_path}...")
    torch.save(std_embed, std_embed_path)

    embed_dict['embeddings'] = embeds
    embed_dict['mean'] = mean_embed
    embed_dict['std'] = std_embed

    if indices is not None:
        indices_path = get_embed_path(embedding_type='indices', **argdict)
        print(f"Saving indices to {indices_path}...")
        torch.save(indices, indices_path)
        embed_dict['indices'] = indices

    embeds_whitened = whiten_embeddings(embeds, mean_embed, std_embed)
    embeds_whitened_path = get_embed_path(
        embedding_type='whitened',
        **argdict,
    )
    print(f"Saving whitened embeddings to {embeds_whitened_path}...")
    torch.save(embeds_whitened, embeds_whitened_path)
    embed_dict['whitened'] = embeds_whitened

    whitened_from_dataset = argdict.get('whitened_from_dataset', None)
    if whitened_from_dataset is not None:
        zscore_argdict = argdict.copy()
        zscore_argdict['dataset'] = whitened_from_dataset
        zscore_mean_path = get_embed_path(**zscore_argdict, embedding_type='mean')
        zscore_std_path = get_embed_path(**zscore_argdict, embedding_type='std')

        zscore_mean_embed = torch.load(zscore_mean_path)
        zscore_std_embed = torch.load(zscore_std_path)

        embeds_cross_whitened = whiten_embeddings(embeds, zscore_mean_embed, zscore_std_embed)
        embeds_cross_whitened_path = get_embed_path(
            embedding_type='whitened',
            cross_whiten=True,
            **argdict,
        )

        print(f"Saving cross-whitened embeddings to {embeds_cross_whitened_path}...")
        torch.save(embeds_cross_whitened, embeds_cross_whitened_path)
        embed_dict['cross_whitened'] = embeds_cross_whitened

    return embed_dict

def main():
    args = parse_args()
    argdict = vars(args)
    cache_embeddings(argdict)

if __name__ == "__main__":
    main()