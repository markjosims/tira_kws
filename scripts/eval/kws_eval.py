"""
Given a KWS list mapping keyphrases to positive and negative records,
compute ROC AUC scores for keyphrase spotting using embeddings from
the given dataset and encoder.
"""

import json

import numpy as np

from src.constants import (
    KWS_PREDICTIONS, KEYPHRASE_LIST,
    CALIBRATION_LIST, ENGLISH_CALIBRATION_LIST,
    DEVICE, CONFIG_DIR
)
from src.encoding import get_cosine_similarity, pool_embeds
from src.wfst import decode_embed_list
from src.dtw import pairwise_dtw
from scripts.cache_embeddings import add_cache_embeddings_args, load_embeddings
import pandas as pd
import torch
from torchmetrics.classification import BinaryEER, BinaryROC
from torchmetrics.retrieval import (
    RetrievalMAP, RetrievalRecall, RetrievalAUROC, RetrievalHitRate
)
from typing import *
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
import hydra
import os
from tqdm import tqdm

load_dotenv()
EXPERIMENT_NAME_TEMPLATE = "tira_kws_auroc_ws_{encoder}_{window_size}"

# easy, medium and hard refer to different splits of Tira keyphrases
# based on string similarity with the keyphrase query, with dissimilar
# words in easy and similar words in hard
# English indicates Tira/English LID, and is not binned by difficulty
CASES = ['easy', 'medium', 'hard', 'english']
METRIC_FUNCTS = {
    'auroc': RetrievalAUROC,
    'eer': BinaryEER,
    'mAP': RetrievalMAP,
    'recall': RetrievalRecall,
    'hit_rate': RetrievalHitRate,
}
NON_RETRIEVAL_METRICS = ['eer']
ROC_METRICS = {}
K_VALS = [1, 5, 10]

METRICS = {}

for case in CASES:
    ROC_METRICS[case] = BinaryROC()
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
    hit_scores = torch.cat([positive_scores, negative_scores])
    indices = torch.full_like(hit_scores, index, dtype=int)

    # compute single-number metrics
    for metric_name, metric_funct in METRICS[case].items():
        if metric_name in NON_RETRIEVAL_METRICS:
            metric_args = (hit_scores, labels)
        else:
            metric_args = (hit_scores, labels, indices)
        batch_metrics[f"{case}/{metric_name}"] = metric_funct(*metric_args)

    # compute ROC values and thresholds
    fpr, tpr, thresholds = ROC_METRICS[case](hit_scores, labels)
    fpr_tpr_diff = fpr - (1-tpr)
    fpr_tpr_diff = fpr_tpr_diff.abs()
    min_diff_idx = fpr_tpr_diff.argmin()
    eer_threshold = thresholds[min_diff_idx].item()

    batch_metrics[f"{case}/eer_threshold"] = eer_threshold

    return batch_metrics

def index_embeddings_safe(
        embeddings: Union[torch.Tensor, Sequence[torch.Tensor]],
        indices: Sequence[int],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Indexes embedding rows from either a torch tensor or list/tuple
    of torch tensors.

    Args:
        embeddings: 2D torch.Tensor of shape (batch_size, embed_dim)
            or list of torch tensors, each of shape (num_windows, embed_dim)
        indices: indices of records to index from embeddings

    Returns:
        tensor or list of tensors indexed from `embeddings`
    """
    if type(embeddings) is torch.Tensor:
        return embeddings[indices]
    return [embeddings[i] for i in indices]

def evaluate_keyphrase_batch(
        tira_embeddings: torch.Tensor,
        eng_embeddings: torch.Tensor,
        keyphrase_object: Dict[str, Any],
        cfg: Dict[str, Any],
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
        cfg: global config

    Returns:
        List of dicts with keyphrases and metric values
    """
    positive_idcs = keyphrase_object['record_idcs']
    keyphrase_idx = keyphrase_object['keyphrase_idx']

    similarity_funct = get_similarity_funct(tira_embeddings, cfg)

    # calculate similarity matrices w/in positive embeds
    # between positive and negative Tira embeds
    # (split between easy medium and hard)
    # and between positive Tira embeds and negative English embeds

    positive_embeds = index_embeddings_safe(tira_embeddings, positive_idcs)
    positive_similarity = similarity_funct(positive_embeds, positive_embeds)
    # ignore self-similarity
    positive_similarity.fill_diagonal_(torch.nan)

    negative_scores = {}

    for case in CASES:
        if case == 'english':
            negative_embeds = eng_embeddings
        else:
            negative_idcs = keyphrase_object[case]
            negative_embeds = index_embeddings_safe(tira_embeddings, negative_idcs)

        if len(negative_embeds) == 0:
            # some records may not have data for a particular case
            continue
        negative_scores[case] = similarity_funct(positive_embeds, negative_embeds)
    
    # now iterate through each positive embedding, index the appropriate scores
    # and pass to `evaluate_keyphrase`
    batch_results = []
    for i, positive_scores in enumerate(positive_similarity):
        positive_scores_for_record = positive_scores[~torch.isnan(positive_scores)]
        negative_scores_for_record = {}
        for case, scores in negative_scores.items():
            negative_scores_for_record[case] = scores[i]
        batch_results.append(evaluate_keyphrase(
            positive_scores_for_record,
            negative_scores_for_record,
            keyphrase_object,
        ))
    return batch_results


def get_similarity_funct(
        tira_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        cfg: Dict[str, Any],
) -> Callable[..., torch.Tensor]:
    """
    Loads similarity function specified in global config
    and asserts `tira_embeddings` has appropriate shape

    Args:
        tira_embeddings: tensor or list of tensor of embeddings
        cfg: global config

    Returns:
        similarity_funct: callable that takes two embeddings
            or lists of embeddings as input and returns a
            similarity score

    """
    window_size = cfg.get('window_size', None)
    pooling_type = cfg.get('pooling_type', None)
    windowed_inference_type = cfg.get('windowed_inference_type', 'wfst')
    similarity_funct = None

    # cases where we're comparing single embeddings for each query/test phrase
    if (window_size is None) and (pooling_type is None) and type(tira_embeddings) is not torch.Tensor:
        # we're using a Wav2Vec2 model and using DTW or WFST decoding
        assert windowed_inference_type is not None
    elif (window_size is None) and (pooling_type is None):
        # we're using a variable-input encoder (e.g. CLAP-IPA or Speechbrain)
        # without windows
        assert tira_embeddings.dim() == 2
        similarity_funct = get_cosine_similarity
    elif pooling_type is not None:
        # we're using a Wav2Vec2 model with embedding pooling
        pool_wrapper = lambda t1, t2: (
            pool_embeds(t1, pooling_type),
            pool_embeds(t2, pooling_type)
        )
        similarity_funct = lambda t1, t2: get_cosine_similarity(
            *pool_wrapper(t1, t2)
        )

    if similarity_funct:
        return similarity_funct

    # we're doing DTW or WFST decoding over sequences of embeddings
    assert type(tira_embeddings) in [list, tuple]
    assert tira_embeddings[0].dim() == 2

    if windowed_inference_type == 'wfst':
        # # `decode_embed_list` returns both scores and labels
        # # we just want scores here
        similarity_funct = lambda *args: decode_embed_list(*args)[0]
    elif windowed_inference_type == 'dtw':
        similarity_funct = pairwise_dtw
    elif windowed_inference_type is not None:
        raise ValueError(f"Unknown windowed inference type {windowed_inference_type}")
    else:
        raise ValueError("Must use windowed inference type when using lists of embeddings as input")

    return similarity_funct


def evaluate_keyphrase(
        positive_scores: torch.Tensor,
        negative_scores: Dict[str, torch.Tensor],
        keyphrase_object: Dict[str, Any],
) -> Dict[str, float]:
    """
    Get ROC AUC and EER for a single keyphrase, divided into easy, medium and hard
    for Tira>Tira KWS and a single score for Tira>English KWS.

    Args:
        positive_scores: similarity scores for query embedding vs. other positive keyphrases
        negative_scores: dict containing similarity score tensors for query embedding vs.
            negative embeddings for all four cases (easy, medium and hard Tira, and English)
        keyphrase_object: Dict with metadata for current keyphrase
    Returns:
        List of dicts with keyphrases and metric values
    """
    result = dict(
        keyphrase=keyphrase_object['keyphrase'],
        keyphrase_idx=keyphrase_object['keyphrase_idx'],
    )

    for case in CASES:
        if case not in negative_scores:
            continue
        negative_scores_for_case = negative_scores[case]
        metrics_for_case = compute_metrics(
            positive_scores,
            negative_scores_for_case,
            case,
            keyphrase_object['keyphrase_idx'],
        )
        result.update(**metrics_for_case)
    return result


def compute_roc_auc(cfg, run=None) -> pd.DataFrame:
    tira_argdict = cfg.copy()
    eng_argdict = cfg.copy()

    tira_argdict['dataset'] = getattr(cfg, 'dataset', 'tira_asr')
    eng_argdict['dataset'] = getattr(cfg, 'drz_dataset', 'tira_drz')
    # ignore whitening for now
    # TODO: define subset of embeddings to calculate z-score from
    # eng_argdict['whitened_from_dataset'] = args.tira

    tira_embed_dict = load_embeddings(tira_argdict)
    eng_embed_dict = load_embeddings(eng_argdict)

    tira_embeds = tira_embed_dict['embeddings']
    eng_embeds = eng_embed_dict['embeddings']

    kws_list = load_kws_list(cfg)
    eng_idcs = load_eng_list(cfg)

    if eng_idcs is not None:
        eng_embeds = index_embeddings_safe(eng_embeds, eng_idcs)

    results = []
    for keyphrase_object in tqdm(kws_list, desc='Evaluating keyphrases...'):
        keyphrase_results = evaluate_keyphrase_batch(
            tira_embeds,
            eng_embeds,
            keyphrase_object,
            cfg,
        )
        results.extend(keyphrase_results)

    df = pd.DataFrame(results)
    return df

def load_eng_list(args) -> Optional[List[int]]:
    if args.list_type == 'all':
        return None
    # load english calibration keyphrase indices
    with open(ENGLISH_CALIBRATION_LIST, 'r') as f:
        english_keyphrase_idcs = [int(line.strip()) for line in f]
    return english_keyphrase_idcs

def load_kws_list(args):
    list_path = get_list_path(args)
    with open(list_path, 'r') as f:
        kws_list = json.load(f)
    return kws_list

def get_list_path(args):
    if args.list_type == 'all':
        list_path = KEYPHRASE_LIST
    elif args.list_type == 'calibrate':
        list_path = CALIBRATION_LIST
    else:
        raise ValueError(f"Unknown list type: {args.list_type}")
    return list_path

@hydra.main(
    version_base="1.3",
    config_path=str(CONFIG_DIR),
    config_name='default')
def main(cfg: DictConfig) -> int:
    with wandb.init(
        project=os.environ.get('WANDB_PROJECT', None),
        entity=os.environ.get('WANDB_ENTITY', None),
    ) as run:
        experiment_name = get_experiment_name(cfg)
        run.name = experiment_name

        cfg_container = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False,
        )
        run.config.update(cfg_container)

        df = compute_roc_auc(cfg, run=run)

        run.log(data={"kws_eval_results": wandb.Table(dataframe=df)})
        for case, case_metrics in METRICS.items():
            print(f"Computing overall metrics for case {case}...")
            for metric_name, metric_obj in case_metrics.items():
                run.summary[f"{case}/{metric_name}"] = metric_obj.compute()
            run.summary[f"{case}/mean_eer_threshold"] = df[f'{case}/eer_threshold'].mean()
            run.summary[f"{case}/std_eer_threshold"] = df[f'{case}/eer_threshold'].std()

            print(f"Drawing ROC plot for case {case}...")
            roc_funct = ROC_METRICS[case]
            _, ax = roc_funct.plot(score=True)
            run.log({f"{case}/roc_curve": ax})
                

    output_file = KWS_PREDICTIONS / f"{experiment_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved KWS eval results to {output_file}")

    return 0

def get_experiment_name(cfg):
    window_size_str = 'whole_utterance'
    if hasattr(cfg, 'window_size'):
        window_size_str = str(cfg.window_size).replace('.', 'p')
        window_size_str = 'ws' + window_size_str

    encoder_str = cfg.encoder
    if cfg.encoder_size is not None:
        # not all encoders have sizes
        encoder_str += f"_{cfg.encoder_size}"
    experiment_name = EXPERIMENT_NAME_TEMPLATE.format(
        encoder=f"{cfg.encoder}_{cfg.encoder_size}",
        window_size=window_size_str,
    )
    return experiment_name

if __name__ == '__main__':
    main()