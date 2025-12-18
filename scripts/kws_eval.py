"""
Given a KWS list mapping keyphrases to positive and negative records,
compute ROC AUC scores for keyphrase spotting using embeddings from
the given dataset and encoder.
"""

from argparse import ArgumentParser
import json
from constants import (
    KWS_PREDICTIONS, KEYPHRASE_LIST,
    CALIBRATION_LIST, ENGLISH_CALIBRATION_LIST,
    DEVICE
)
from encoding import get_cosine_distance
from wfst import decode_embed_list
from scripts.cache_embeddings import add_cache_embeddings_args, load_embeddings
import pandas as pd
import torch
from torchmetrics.classification import BinaryEER
from torchmetrics.retrieval import RetrievalMAP, RetrievalRecall, RetrievalAUROC
from typing import *
import wandb
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()
EXPERIMENT_NAME_TEMPLATE = "tira_kws_auroc_ws_{encoder}_{window_size}"

# easy, medium and hard refer to different splits of Tira keyphrases
# based on string distance with the keyphrase query, with dissimilar
# words in easy and similar words in hard
# English indicates Tira/English LID, and is not binned by difficulty
CASES = ['easy', 'medium', 'hard', 'english']
METRIC_FUNCTS = {
    'auroc': RetrievalAUROC,
    'eer': BinaryEER,
    'mAP': RetrievalMAP,
    'recall': RetrievalRecall,
}
NON_RETRIEVAL_METRICS = ['eer']
K_VALS = [1, 5, 10]

METRICS = {}

for case in CASES:
    METRICS[case]={}
    for metric in METRIC_FUNCTS:
        METRICS[case][metric]=METRIC_FUNCTS[metric]().to(DEVICE)
        # retrieval metrics calculated for various values of `k`
        if metric not in NON_RETRIEVAL_METRICS:
            for k in K_VALS:
                METRICS[case][f"{metric}@{k}"]\
                    =METRIC_FUNCTS[metric](top_k=k)\
                    .to(DEVICE)

def compute_metrics(
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        case: str,
        index: int,
) -> Dict[str, float]:
    """
    Compute AUROC, EER, mAP and recall given positive and negative scores.

    Args:
        positive_scores: Tensor of shape (num_positive,)
        negative_scores: Tensor of shape (num_negative,)
        case: str indicating case being evaluated
        index: int indicating index of keyphrase being evaluated
    Returns:
        Dict with metric values
    """
    batch_metrics = {}
    if negative_scores.shape[0] == 0:
        for metric in METRICS[case]:
            batch_metrics[f"{case}/{metric}"] = torch.nan
        return batch_metrics
    
    labels = torch.cat([
        torch.ones_like(positive_scores),
        torch.zeros_like(negative_scores),
    ]).to(int)
    distance_scores = torch.cat([positive_scores, negative_scores])
    hit_scores = 1-distance_scores
    indices = torch.full_like(hit_scores, index, dtype=int)
    
    for metric_name, metric_funct in METRICS[case].items():
        if metric_name in NON_RETRIEVAL_METRICS:
            metric_args = (hit_scores, labels)
        else:
            metric_args = (hit_scores, labels, indices)
        batch_metrics[f"{case}/{metric_name}"] = metric_funct(*metric_args)
    return batch_metrics

def evaluate_keyphrase_batch(
        tira_embeddings: torch.Tensor,
        eng_embeddings: torch.Tensor,
        keyphrase_object: Dict,
) -> List[Dict[str, float]]:
    """
    Get evaluation metrics for all tokens of the keyphrase indicates by `keyphrase_object`.
    Wraps `evaluate_keyphrase`, which gets metrics for a single token.

    Retrieves positive Tira embeddings using the 'record_idcs' key of `keyphrase_object`
    and negative embeddings from 'easy', 'medium' and 'hard' keys.
    Assumes `eng_embeddings` already contains only the relevant embeddings used for KWS
    evaluation.

    Args:
        tira_embeddings: Tensor of shape (tira_corpus_size, embed_dim)
        eng_embeddings: Tensor of shape (eng_corpus_size, embed_dim)
        keyphrase_object: Dict with keys 'keyphrase', 'keyphrase_idx', 'record_idcs',
            'easy', 'medium', 'hard'

    Returns:
        List of dicts with keyphrases and metric values
    """
    positive_idcs = keyphrase_object['record_idcs']
    keyphrase_idx = keyphrase_object['keyphrase_idx']

    # a 2d embedding matrix indicates non-windowed embeddings
    # and a 3d matrix indicates windowed
    if len(tira_embeddings.shape) == 2:
        distance_funct = get_cosine_distance
    elif len(tira_embeddings.shape) == 3:
        # `decode_embed_list` returns both scores and labels
        # we just want scores here
        distance_funct = lambda *args: decode_embed_list(*args)[0]
    else:
        raise ValueError(f"Invalid embedding dimension {tira_embeddings.shape}, must be 2 or 3")

    # calculate distance matrices w/in positive embeds
    # between positive and negative Tira embeds
    # (split between easy medium and hard)
    # and between positive Tira embeds and negative English embeds
    positive_embeds = tira_embeddings[positive_idcs]
    positive_distance = distance_funct(positive_embeds, positive_embeds)
    # ignore self-distance
    positive_distance.fill_diagonal_(torch.nan)

    negative_scores = {}

    for case in CASES:
        if case == 'english':
            negative_embeds = eng_embeddings
        else:
            negative_idcs = keyphrase_object[case]
            negative_embeds = tira_embeddings[negative_idcs]
        negative_scores[case] = distance_funct(positive_embeds, negative_embeds)
    
    # now iterate through each positive embedding, index the appropriate scores
    # and pass to `evaluate_keyphrase`
    for i, positive_scores in enumerate(positive_distance):
        positive_scores = positive_scores[~torch.isnan(positive_scores)]


def evaluate_keyphrase(
        tira_tira_scores: torch.Tensor,
        tira_eng_scores: torch.Tensor,
        keyphrase_object: Dict[str, Any],
) -> Dict[str, float]:
    """
    TODO: should take `positive_scores` and `negative_scores` as args
    where `negative_scores` is a dict of score tensors for each case

    Get ROC AUC and EER for a single keyphrase, divided into easy, medium and hard
    for Tira>Tira KWS and a single score for Tira>English KWS.

    Args:
        tira_tira_scores: Tensor of shape (num_tira_records, num_tira_records)
        tira_eng_scores: Tensor of shape (num_tira_records, num_eng_records)
        keyphrase_object: Dict with keys 'keyphrase', 'keyphrase_idx', 'record_idcs',
            'easy', 'medium', 'hard'
    Returns:
        List of dicts with keyphrases and metric values
    """
    positive_idcs = keyphrase_object['record_idcs']
    keyphrase_idx = keyphrase_object['keyphrase_idx']
    results = []
    for query_idx in positive_idcs:
        batch_positive_idcs = torch.tensor(positive_idcs)
        batch_positive_idcs = batch_positive_idcs[batch_positive_idcs != query_idx]
        batch_result = dict(keyphrase=keyphrase_object['keyphrase'], query_idx=query_idx)

        positive_scores = tira_tira_scores[query_idx, batch_positive_idcs]
        for case in CASES:
            
            if case == 'english':
                negative_scores = tira_eng_scores[query_idx]
            else:
                negative_idcs = keyphrase_object[case]
                negative_scores = tira_tira_scores[query_idx, negative_idcs]
            batch_metrics = compute_metrics(
                positive_scores,
                negative_scores,
                case,
                keyphrase_idx,
            )
            batch_result.update(**batch_metrics)
        results.append(batch_result)
    return results


def compute_roc_auc(args, run=None) -> pd.DataFrame:
    tira_argdict = vars(args).copy()
    eng_argdict = vars(args).copy()

    tira_argdict['dataset'] = args.dataset
    eng_argdict['dataset'] = args.drz_dataset
    # ignore whitening for now
    # TODO: define subset of embeddings to calculate z-score from
    # eng_argdict['whitened_from_dataset'] = args.tira

    tira_embed_dict = load_embeddings(tira_argdict)
    eng_embed_dict = load_embeddings(eng_argdict)

    tira_embeds = tira_embed_dict['embeddings']
    eng_embeds = eng_embed_dict['embeddings']

    kws_list = load_kws_list(args)
    eng_idcs = load_eng_list(args)

    if eng_idcs is not None:
        eng_embeds = eng_embeds[eng_idcs]

    tira_tira_distance = get_cosine_distance(tira_embeds, tira_embeds)
    tira_eng_distance = get_cosine_distance(tira_embeds, eng_embeds)

    results = []
    for keyphrase_object in tqdm(kws_list, desc='Evaluating keyphrases...'):
        keyphrase_results = evaluate_keyphrase(
            tira_tira_scores=tira_tira_distance,
            tira_eng_scores=tira_eng_distance,
            keyphrase_object=keyphrase_object,
        )
        results.extend(keyphrase_results)

    df = pd.DataFrame(results)
    return df


def load_eng_list(args):
    if args.list_type == 'all':
        return None
    # load english calibration keyphrase indices
    with open(ENGLISH_CALIBRATION_LIST, 'r') as f:
        english_keyphrase_idcs = set([int(line.strip()) for line in f])
    return english_keyphrase_idcs

def load_kws_list(args):
    list_path = get_list_path(args)
    with open(list_path, 'r') as f:
        kws_list = json.load(f)
    return kws_list

def get_list_path(args):
    if args.list_type == 'all':
        list_path = KEYPHRASE_LIST
    elif args.list_type == 'calibrated':
        list_path = CALIBRATION_LIST
    else:
        raise ValueError(f"Unknown list type: {args.list_type}")
    return list_path

def main():
    args = parse_args()
    with wandb.init(
        project=os.environ.get('WANDB_PROJECT', None),
        entity=os.environ.get('WANDB_ENTITY', None),
    ) as run:
        experiment_name = get_experiment_name(args)
        run.name = experiment_name
        run.config.update(vars(args))

        df = compute_roc_auc(args, run=run)

        run.log({"roc_auc_results": wandb.Table(dataframe=df)})
        for case, case_metrics in METRICS.items():
            print(f"Computing overall metrics for case {case}...")
            for metric_name, metric_obj in case_metrics.items():
                run.summary[f"{case}/{metric_name}"] = metric_obj.compute()
                

    output_file = KWS_PREDICTIONS / f"{experiment_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved KWS ROC AUC results to {output_file}")

def parse_args():
    parser = ArgumentParser(description="Compute AUROC for Tira KWS")
    parser = add_cache_embeddings_args(parser)
    parser.add_argument('--drz-dataset', type=str, default='tira_drz')
    parser.add_argument(
        "--list_type", '-t', type=str, default='all',
        choices=['all', 'calibrated'],
        help='KWS list to use for evaluation.',
    )
    args = parser.parse_args()
    return args

def get_experiment_name(args):
    window_size_str = 'whole_utterance'
    if args.window_size is not None:
        window_size_str = str(args.window_size).replace('.', 'p')
        window_size_str = '_ws' + window_size_str

    encoder_str = args.encoder
    if args.encoder_size is not None:
        # not all encoders have sizes
        encoder_str += f"_{args.encoder_size}"
    experiment_name = EXPERIMENT_NAME_TEMPLATE.format(
        encoder=f"_{args.encoder}_{args.encoder_size}",
        window_size=window_size_str,
    )
    return experiment_name

if __name__ == '__main__':
    main()