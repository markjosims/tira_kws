"""
Script to cache speech embeddings for a dataset. Saves embeddings, mean embedding, and std embedding to disk.
Also saves whitened embeddings (z-score normalized) to disk. If a z-score dataset is provided to
--whitened_from_dataset, also saves embeddings whitened using the mean and std from that dataset.

Arguments:
    --dataset: Name of the dataset to process.
    --encoder: Encoder model to use for generating embeddings.
    --encoder_size: Size of the encoder model.
    --whitened_from_dataset: (Optional) Name of dataset to use for z-score normalization.
"""

from argparse import ArgumentParser
import os

from torch.nn.utils.rnn import pad_sequence

from src.constants import EMBEDDINGS
from src.dataloading import load_dataset, prepare_dataset, add_dataset_args
from distance import add_sliding_window_args, add_encoder_args
import torch
from typing import *

def add_cache_embeddings_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Adds arguments relevant for caching embeddings to the given ArgumentParser.
    Adds arg groups for dataset, sliding window, and encoder arguments, and adds
    the --whitened_from_dataset/-z argument indicating the dataset from which to
    normalize the current dataset's embeddings using the z-score.
    """
    parser = add_dataset_args(parser)
    parser = add_sliding_window_args(parser)
    parser.add_argument('--whitened_from_dataset', '-z', type=str, default=None, help='Name of dataset to use for z-score normalization.')
    parser = add_encoder_args(parser)
    return parser


def parse_args():
    parser = ArgumentParser(description="Compute embeddings for a dataset and cache them to disk.")
    parser = add_cache_embeddings_args(parser)
    return parser.parse_args()

def get_embed_path(
        dataset: str,
        encoder: str,
        encoder_size: str,
        window_size: Optional[float] = None,
        embedding_type: Literal['regular', 'mean', 'std', 'whitened'] = 'regular',
        whitened_from_dataset: Optional[str] = None,
        cross_whiten: bool = False,
        num_records: Optional[int] = None,
        **_, # to allow for unused args
) -> str:
    
    if whitened_from_dataset is not None and cross_whiten:
        embed_stem = f"whitened_from_{whitened_from_dataset}_embeddings"
    elif embedding_type == 'whitened':
        embed_stem = "whitened_embeddings"
    elif embedding_type != 'regular':
        embed_stem = f"{embedding_type}_embedding"
    else:
        embed_stem = "embeddings"

    embed_dir = EMBEDDINGS / f"{dataset}_{encoder}_{encoder_size}"

    if window_size is not None:
        window_size_str = str(window_size).replace('.', 'p')
        embed_stem += f"_ws{window_size_str}"
    if num_records is not None:
        embed_stem += f"_nr{num_records}"

    embed_path = embed_dir / f"{embed_stem}.pt"
        
    return str(embed_path)

def get_all_embed_paths(argdict) -> Dict[str, str]:
    """
    Returns a dictionary of all relevant embedding paths for the given argument dictionary.
    """
    paths = {
        'embeddings': get_embed_path(**argdict, embedding_type='regular'),
        'mean': get_embed_path(**argdict, embedding_type='mean'),
        'std': get_embed_path(**argdict, embedding_type='std'),
        'whitened': get_embed_path(**argdict, embedding_type='whitened')
    }
    whitened_from_dataset = argdict.get('whitened_from_dataset', None)
    if whitened_from_dataset is not None:
        paths['cross_whitened'] = get_embed_path(
            **argdict,
            embedding_type='whitened',
            cross_whiten=True,
        )
    return paths

def load_embeddings(
        argdict: Dict[str, Any],
        compute_if_not_found: bool = True,
        list_to_padded_tensor: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Loads all relevant embeddings for the given argument dictionary.
    If any embeddings are missing, computes and caches them first.

    Args:
        argdict (Dict[str, Any]): Dictionary of arguments including dataset, encoder, etc.  
        compute_if_not_found (bool): Whether to compute embeddings if not found on disk.
        list_to_padded_tensor (bool): Whether to cast lists of windowed embeddings to a single
            zero-padded tensor
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing loaded embeddings.
    """
    embed_dict = {}
    paths = get_all_embed_paths(argdict)
    embeds_exist = all(os.path.exists(path) for path in paths.values())
    num_records = argdict.get('num_records', None)
    if not embeds_exist and num_records is not None:
        argdict_no_nr = argdict.copy()
        argdict_no_nr['num_records'] = None
        embeds_no_nr = load_embeddings(
            argdict_no_nr,
            compute_if_not_found=False,
            list_to_padded_tensor=list_to_padded_tensor,
        )
        if embeds_no_nr is not None:
            print("Loading embeddings computed without num_records restriction...")
            for key in paths.keys():
                embed_dict[key] = embeds_no_nr[key][:num_records]
            return embed_dict

    if not embeds_exist and compute_if_not_found:
        print("Embedding file not found, computing embeddings and caching...")
        embed_dict = cache_embeddings(argdict)
    elif not embeds_exist:
        raise FileNotFoundError("Embedding file not found")
    else:
        print("Loading embeddings from disk...")
        for key, path in paths.items():
            embed_dict[key] = torch.load(path)

    if list_to_padded_tensor:
        for key, embed in embed_dict.items():
            if type(embed) in (list, tuple):
                embed_dict[key] = pad_sequence(embed, batch_first=True, padding_value=0.0)

    return embed_dict


def compute_embeddings(
        dataset: str = None,
        num_records: Optional[int] = None,
        encoder: str = None,
        encoder_size: str = None,
        window_size: Optional[float] = None,
        window_hop: Optional[float] = None,
        **_, # to allow for unused args
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
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
    embeds: torch.Tensor = ds_encoded['input_features'][:]
    mean_embed = embeds.mean(dim=0)
    std_embed = embeds.std(dim=0)

    if (window_size is not None) or (encoder in ['xlsr', 'wav_lm']):
        # transform embeds into a tuple of tensors
        # where each tensor is a sequence of window embeddings
        # first get a tensor indicating which indices begin a new record
        indices = ds_encoded['index'][:]
        start_boundary = torch.tensor([1], device=indices.device)
        boundary_indicators = indices.diff().ne(0)
        boundary_indicators = torch.cat([start_boundary, boundary_indicators])
        start_indices = torch.where(boundary_indicators)[0]

        # now we can use this to get a tensor indicating the number of windows
        # per record
        last_record_len = torch.tensor([embeds.shape[0] - start_indices[-1]])
        record_lens = start_indices.diff()
        record_lens = torch.concat([record_lens, last_record_len])

        # we then use `record_lens` to split the embeds tensor
        # into a tuple of tensors
        embeds = torch.split(embeds, record_lens.tolist(), dim=0)

    return embeds, mean_embed, std_embed

def whiten_embeddings(
        embeds: Union[torch.Tensor, Sequence[torch.Tensor]],
        mean_embed: torch.Tensor,
        std_embed: torch.Tensor
) -> torch.Tensor:
    """
    Z-score normalizes the embeddings using the provided mean and std embeddings.
    """
    if type(embeds) is torch.Tensor:
        return (embeds - mean_embed) / std_embed
    return tuple(whiten_embeddings(record, mean_embed, std_embed) for record in embeds)


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

    embeds, mean_embed, std_embed = compute_embeddings(**argdict)
    embeds_path = get_embed_path(**argdict)
    mean_embed_path = get_embed_path(
        embedding_type='mean',
        **argdict,
    )
    std_embed_path = get_embed_path(
        embedding_type='std',
        **argdict,
    )

    embed_dir = os.path.dirname(mean_embed_path)
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir, exist_ok=True)

    print(f"Saving embeddings to {embeds_path}...")
    torch.save(embeds, embeds_path)
    print(f"Saving mean embedding to {mean_embed_path}...")
    torch.save(mean_embed, mean_embed_path)
    print(f"Saving std embedding to {std_embed_path}...")
    torch.save(std_embed, std_embed_path)

    embed_dict['embeddings'] = embeds
    embed_dict['mean'] = mean_embed
    embed_dict['std'] = std_embed

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