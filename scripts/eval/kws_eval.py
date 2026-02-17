"""
Reads DTW scores from a manifest file under data/distance and evaluates
the KWS performance against the reference labels in keyword_sentence_ids.csv.
Measure precision, recall, and F1 score at different k values, as well as
AUROC and AUPRC.
"""

import torch
from tira_kws.constants import DISTANCE_DIR, KEYWORD_LIST, KEYWORD_SENTENCES
from torchmetrics.classification import BinaryEER, BinaryROC, F1Score
from torchmetrics.retrieval import (
    RetrievalMAP, RetrievalRecall, RetrievalAUROC, RetrievalHitRate
)
import argparse
import pandas as pd
import json
from tqdm import tqdm

def main():
    args = get_args()
    feature_name = args.feature_name

    print("Instantiating metrics...")
    metric_dict = {
        'eer': BinaryEER(),
        'roc': BinaryROC(),
        'f1_macro': F1Score(task='binary', average='macro'),
        'f1_micro': F1Score(task='binary', average='micro'),
    }
    for k in args.k_values:
        metric_dict[f'map@{k}'] = RetrievalMAP(top_k=k)
        metric_dict[f'recall@{k}'] = RetrievalRecall(top_k=k)
        metric_dict[f'hit_rate@{k}'] = RetrievalHitRate(top_k=k)
        metric_dict[f'auroc@{k}'] = RetrievalAUROC(top_k=k)
    

    print("Loading DTW scores and reference labels...")
    dtw_manifest = pd.read_csv(DISTANCE_DIR / feature_name / "manifest_dtw.csv")

    with open(KEYWORD_LIST) as f:
        keywords = json.load(f)

    eval_results = []

    for keyword_obj in tqdm(keywords):
        keyword_index = keyword_obj['word_idx']

        positive_record_idcs = keyword_obj['positive_record_idcs']
        for positive_record_idx in positive_record_idcs:

            positive_record_mask = dtw_manifest['positive_record_id'] == positive_record_idx
            positive_scores = dtw_manifest[
                positive_record_mask &
                dtw_manifest['test_record_id'].isin(positive_record_idcs)
            ]['dtw_score']
            negative_scores = dtw_manifest[
                positive_record_mask & 
                ~dtw_manifest['test_record_id'].isin(positive_record_idcs)
            ]['dtw_score']

            positive_scores = positive_scores.to_list()
            negative_scores = negative_scores.to_list()
            
            positive_labels = [1] * len(positive_scores)
            negative_labels = [0] * len(negative_scores)

            all_scores = positive_scores + negative_scores
            all_labels = positive_labels + negative_labels
            all_indices = [keyword_index] * len(all_labels)

            all_scores = torch.tensor(all_scores)
            all_labels = torch.tensor(all_labels)
            all_indices = torch.tensor(all_indices)

            # negate scores since DTW scores are distances
            # then apply sigmoid so scores are between 0 and 1
            all_scores = -all_scores
            all_scores = torch.sigmoid(all_scores)

            batch_metrics = compute_metrics(
                all_scores, all_labels, all_indices, metric_dict,
            )
            eval_results.append(batch_metrics)

    summary_metrics = {}
    for metric, metric_obj in metric_dict.items():
        if metric == 'roc':
            continue
        summary_metrics[metric] = metric_obj.compute().item()

    # prepend summary metrics to eval results list so it will be the first row in the output CSV
    eval_results = [summary_metrics] + eval_results
    eval_results_df = pd.DataFrame(eval_results)

    # compute average eer_threshold across all keywords and add to summary metrics
    average_eer_threshold = eval_results_df['eer_threshold'].mean()
    summary_metrics['average_eer_threshold'] = average_eer_threshold
    eval_results_df.at[0, 'average_eer_threshold'] = average_eer_threshold


    print("Evaluation complete. Summary metrics:")
    for metric, value in summary_metrics.items():
        print(f"{metric}: {value}")

    print("Saving evaluation results...")

    output_path = DISTANCE_DIR / feature_name / "evaluation_results.csv"
    eval_results_df.to_csv(output_path, index=False)

def compute_metrics(all_scores, all_labels, all_indices, metric_dict):
    batch_metrics = {}

    # compute EER threshold using ROC curve
    fpr, tpr, thresholds = metric_dict['roc'](all_scores, all_labels)
    fpr_tpr_diff = fpr - (1-tpr)
    fpr_tpr_diff = fpr_tpr_diff.abs()
    min_diff_idx = fpr_tpr_diff.argmin()
    eer_threshold = thresholds[min_diff_idx].item()

    batch_metrics["eer_threshold"] = eer_threshold

    # get predicted labels using EER threshold and compute F1 scores
    pred_labels = (all_scores >= eer_threshold).long()
    batch_metrics["f1_macro"] = metric_dict['f1_macro'](pred_labels, all_labels)
    batch_metrics["f1_micro"] = metric_dict['f1_micro'](pred_labels, all_labels)

    # compute remaining metrics
    for metric_name, metric in metric_dict.items():
        if metric_name in batch_metrics:
            continue
        elif '@' in metric_name:
            batch_metrics[metric_name] = metric(all_scores, all_labels, all_indices)
        else:
            batch_metrics[metric_name] = metric(all_scores, all_labels)

    return batch_metrics

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_name",
        type=str,
        default="xlsr",
        help="Name of the feature set to compute pairwise distances for.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 30, 50],
        help="List of k values to evaluate precision, recall, and F1 score at.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()