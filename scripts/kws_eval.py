"""
Given a KWS list mapping keyphrases to positive and negative records,
compute ROC AUC scores for keyword spotting using embeddings from
the given dataset and encoder.
"""

from argparse import ArgumentParser
import json
from constants import (
    KWS_PREDICTIONS, KEYPHRASE_LIST,
    CALIBRATION_LIST, ENGLISH_CALIBRATION_LIST
)
from scripts.cache_embeddings import add_cache_embeddings_args, load_embeddings
import pandas as pd
import torch
from torchmetrics.classification import BinaryAUROC, BinaryEER, BinaryROC
from typing import *
import wandb
from dotenv import load_dotenv
import os

load_dotenv()
EXPERIMENT_NAME_TEMPLATE = "tira_kws_auroc_ws_{encoder}_{window_size}"

auroc = BinaryAUROC()
eer = BinaryEER()
roc = BinaryROC()

def compute_metrics(
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute ROC AUC and EER given positive and negative scores.

    Args:
        positive_scores: Tensor of shape (num_positive,)
        negative_scores: Tensor of shape (num_negative,)
    Returns:
        Dict with 'auroc' and 'eer' scores
    """
    labels = torch.cat([
        torch.ones_like(positive_scores),
        torch.zeros_like(negative_scores),
    ])
    scores = torch.cat([positive_scores, negative_scores])

    auroc_score = auroc(scores, labels).item()
    eer_score = eer(scores, labels).item()
    return auroc_score, eer_score

def evaluate_keyword(
        tira_tira_scores: torch.Tensor,
        tira_eng_scores: torch.Tensor,
        keyword_object: Dict[str, Any],
        eng_idcs: Optional[Set[int]] = None,
) -> Dict[str, float]:
    """
    Get ROC AUC and EER for a single keyword, divided into easy, medium and hard
    for Tira>Tira KWS and a single score for Tira>English KWS.

    Args:
        tira_tira_scores: Tensor of shape (num_tira_records, num_tira_records)
        tira_eng_scores: Tensor of shape (num_tira_records, num_eng_records)
        keyword_object: Dict with keys 'keyphrase', 'keyphrase_idx', 'record_idcs',
            'easy', 'medium', 'hard'
        eng_idcs: Optional set of English keyphrase indices to use for
            negative examples. If None, use all English keyphrases.
    Returns:
        Dict with AUROC and EER scores
    """
    positive_idcs = keyword_object['record_idcs']
    easy_idcs = keyword_object['easy']
    medium_idcs = keyword_object['medium']
    hard_idcs = keyword_object['hard']
    if eng_idcs is None:
        eng_idcs = torch.arange(tira_eng_scores.shape[1])
    results = []
    for query_idx in positive_idcs:
        batch_positive_idcs = torch.tensor(positive_idcs)
        batch_positive_idcs = batch_positive_idcs[batch_positive_idcs != query_idx]
        batch_result = dict(keyphrase=keyword_object['keyphrase'], query_idx=query_idx)

        # Tira > Tira
        for negative_idcs, difficulty in [
            (easy_idcs, 'easy'), (medium_idcs, 'medium'), (hard_idcs, 'hard')
        ]:
            positive_scores = tira_tira_scores[query_idx, batch_positive_idcs]
            negative_scores = tira_tira_scores[query_idx, negative_idcs]
            auroc_score, eer_score = compute_metrics(positive_scores, negative_scores)
            batch_result[f'{difficulty}/auroc'] = auroc_score
            batch_result[f'{difficulty}/eer'] = eer_score
        # Tira > English
        positive_scores = tira_tira_scores[query_idx, batch_positive_idcs]
        negative_scores = tira_eng_scores[query_idx, eng_idcs]
        auroc_score, eer_score = compute_metrics(positive_scores, negative_scores)
        batch_result[f'eng/auroc'] = auroc_score
        batch_result[f'eng/eer'] = eer_score
        results.append(batch_result)
    return results

def get_similarity_scores(
        query_embeds: torch.Tensor,
        test_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity scores between query embeddings and test embeddings.

    Args:
        query_embeds: Tensor of shape (num_queries, embed_dim)
        test_embeds: Tensor of shape (num_tests, embed_dim)
    Returns:
        Tensor of shape (num_queries, num_tests) with cosine similarity scores
    """
    query_norm = query_embeds / query_embeds.norm(dim=1, keepdim=True)
    test_norm = test_embeds / test_embeds.norm(dim=1, keepdim=True)
    scores = torch.matmul(query_norm, test_norm.T)
    return scores

def compute_roc_auc(args, run=None) -> pd.DataFrame:
    tira_argdict = vars(args).copy()
    eng_argdict = vars(args).copy()

    tira_argdict['dataset'] = args.tira
    eng_argdict['dataset'] = args.eng
    # ignore whitening for now
    # TODO: define subset of embeddings to calculate z-score from
    # eng_argdict['whitened_from_dataset'] = args.tira

    tira_embed_dict = load_embeddings(tira_argdict)
    eng_embed_dict = load_embeddings(eng_argdict)

    tira_embeds = tira_embed_dict['embeddings']
    eng_embeds = eng_embed_dict['embeddings']

    tira_tira_similarity = get_similarity_scores(tira_embeds, tira_embeds)
    tira_eng_similarity = get_similarity_scores(tira_embeds, eng_embeds)

    kws_list = load_kws_list(args)
    eng_idcs = load_eng_list(args)

    results = []
    for keyword_object in kws_list:
        keyword_results = evaluate_keyword(
            tira_tira_scores=tira_tira_similarity,
            tira_eng_scores=tira_eng_similarity,
            keyword_object=keyword_object,
            eng_idcs=eng_idcs,
        )
        results.extend(keyword_results)
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
    with wandb.init(project=os.environ.get('WANDB_PROJECT', None)) as run:
        experiment_name = get_experiment_name(args)
        run.name = experiment_name
        run.config.update(vars(args))

        df = compute_roc_auc(args, run=run)

    run.log({"roc_auc_results": wandb.Table(dataframe=df)})

    output_file = KWS_PREDICTIONS / f"{experiment_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved KWS ROC AUC results to {output_file}")

def parse_args():
    parser = ArgumentParser(description="Compute AUROC for Tira KWS")
    parser = add_cache_embeddings_args(parser)
    parser.add_argument("--dataset", type=str)
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