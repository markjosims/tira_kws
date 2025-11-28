"""
Given Tira-Tira and Tira-English similarity scores, compute following metrics:
- Tira-Tira ROC curves and AUC scores
- Tira-(Tira+English) ROC curves and AUC scores
Use this to compute overall ROC AUC scores for KWS.
Save JSON file with metrics for each keyphrase, i.e.:
[
    {
        "row": 0,
        "keyphrase": "àpɾí jícə̀lò",
        "tira>tira": {
            'fpr': [0.0, 0.1, 0.2, ...],
            'tpr': [0.0, 0.4, 0.6, ...],
            'auc': 0.85
        }
        "tira>tira/english": {
            'fpr': [0.0, 0.05, 0.15, ...],
            'tpr': [0.0, 0.5, 0.7, ...],
            'auc': 0.90
        }
        "tira>english": {
            'fpr': [0.0, 0.02, 0.1, ...],
            'tpr': [0.0, 0.6, 0.8, ...],
            'auc': 0.95,
        }
    },
    ...

]
Include an entry for average metrics, marked by "keyphrase": "$mean",
and another where ROC/AUC is computed by taking the maximum score across keyphrases,
marked by "keyphrase": "$max".

See this page for more on multi-class ROC AUC computation:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
"""

from argparse import ArgumentParser
import json
import re
from constants import (
    SIMILARITY_MATRIX_PATH, CSV_PATH,
    KEYPHRASE_PATH, MAX_KEYWORD_STR
)
from encoding import add_sliding_window_args
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from typing import *
from tqdm import tqdm

DRZ_MATRIX_PATH_TEMPLATE = 'data/similarity_matrix_{dataset}_{language}_ws{window_size}.pt'
JSON_OUPUT_PATH_TEMPLATE = 'data/roc_auc_ws{window_size}.json'

def parse_args():
    parser = ArgumentParser()
    parser = add_sliding_window_args(parser)
    parser.add_argument('--min_duration', '-m', type=float, help="If not set, use window size - 1sec.")
    parser.add_argument('--max_duration', '-M', type=float, help="If not set, use window size + 1sec.")

    args = parser.parse_args()
    if args.max_duration is None:
        args.max_duration = args.window_size + 1.0
    if args.min_duration is None:
        args.min_duration = max(0.0, args.window_size - 1.0)

    return args
    
def get_drz_matrix_path_with_window_size(window_size: float) -> str:
    dataset = 'tira_drz'
    language = 'eng'
    window_size_str = str(window_size).replace('.', '_')
    return DRZ_MATRIX_PATH_TEMPLATE.format(
        dataset=dataset,
        language=language,
        window_size=window_size_str
    )

def get_output_json_path_with_window_size(window_size: float) -> str:
    window_size_str = str(window_size).replace('.', '_')
    return JSON_OUPUT_PATH_TEMPLATE.format(
        window_size=window_size_str
    )

def main():
    args = parse_args()

    df = pd.read_csv(CSV_PATH, index_col='index')


    # Load similarity matrices
    tira_matrix = torch.load(SIMILARITY_MATRIX_PATH).cpu().numpy()
    drz_matrix_path = get_drz_matrix_path_with_window_size(args.window_size)
    drz_matrix = torch.load(drz_matrix_path).cpu().numpy()

    # filter by duration
    ms_per_sec = 1_000
    duration_sec = df['duration']/ms_per_sec
    duration_mask = duration_sec.between(args.min_duration, args.max_duration).to_numpy()

    # filter out unique keyphrases
    unique_mask = ~df['unique'].to_numpy()

    # apply filters
    mask = duration_mask & unique_mask
    df = df[mask]
    df = df.reset_index(drop=True)
    keyphrase_ids = df['transcription_id'].to_numpy()
    tira_matrix = tira_matrix[mask][:, mask]
    drz_matrix = drz_matrix[mask]

    # Load keyphrases
    with open(KEYPHRASE_PATH, 'r', encoding='utf-8') as f:
        keyphrases = [line.strip() for line in f.readlines()]

    results = get_roc_auc_per_keyword(df, tira_matrix, drz_matrix, keyphrase_ids, keyphrases)
    results.append(get_max_roc_auc(tira_matrix, drz_matrix, keyphrase_ids))

    json_path = get_output_json_path_with_window_size(args.window_size)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Saved results to ", json_path)

def get_roc_auc_per_keyword(df, tira_matrix, drz_matrix, keyphrase_ids, keyphrases):
    keyword_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        keyphrase = row['transcription']
        keyphrase_id = keyphrases.index(keyphrase)
        keyword_mask = keyphrase_ids == keyphrase_id
        
        tira_labels = np.zeros_like(tira_matrix[0])
        tira_labels[keyword_mask]=1
        
        if tira_labels.sum() < 2:
            # not enough positive samples to compute ROC/AUC
            # this can happen after filtering duration
            # since other tokens of the same keyphrase may
            # have been in other duration ranges
            continue
        
        # remove current row
        keyword_probs = tira_matrix[i]
        current_row_mask = np.ones_like(tira_labels, dtype=bool)
        current_row_mask[i] = False
        keyword_probs = keyword_probs[current_row_mask]
        tira_labels = tira_labels[current_row_mask]

        # monolingual ROC/AUC
        tira_fpr, tira_tpr, tira_thresholds = roc_curve(
            tira_labels,
            keyword_probs,
        )
        tira_auc = roc_auc_score(tira_labels, keyword_probs)
        
        # bilingual ROC/AUC
        drz_probs = drz_matrix[i]
        drz_labels = np.zeros_like(drz_probs)

        # bilingual ROC/AUC
        all_probs = np.concatenate([keyword_probs, drz_probs])
        all_labels = np.concatenate([tira_labels, drz_labels])

        all_fpr, all_tpr, all_thresholds = roc_curve(
            all_labels,
            all_probs,
        )
        all_auc = roc_auc_score(all_labels, all_probs)

        # english ROC/AUC
        tira_positive_mask = tira_labels.astype(bool)
        tira_positive_labels = tira_labels[tira_positive_mask]
        tira_positive_probs = keyword_probs[tira_positive_mask]

        eng_probs = np.concatenate([tira_positive_probs, drz_probs])
        eng_labels = np.concatenate([tira_positive_labels, drz_labels])
        eng_fpr, eng_tpr, eng_thresholds = roc_curve(
            eng_labels,
            eng_probs,
        )
        eng_auc = roc_auc_score(eng_labels, eng_probs)

        keyword_results.append({
            'row': int(i),
            'keyphrase': keyphrase,
            'tira>tira': {
                'fpr': tira_fpr.tolist(),
                'tpr': tira_tpr.tolist(),
                'auc': float(tira_auc),
            },
            'tira>english': {
                'fpr': eng_fpr.tolist(),
                'tpr': eng_tpr.tolist(),
                'auc': float(eng_auc),
            },
            'tira>tira/english': {
                'fpr': all_fpr.tolist(),
                'tpr': all_tpr.tolist(),
                'auc': float(all_auc),
            }
        })
        
    return keyword_results

def get_max_roc_auc(tira_matrix, drz_matrix, keyphrase_ids):
    tira_matrix = tira_matrix.copy()
    np.fill_diagonal(tira_matrix, -np.inf)
    predicted_row_ids = tira_matrix.argmax(axis=0)
    predicted_keyphrase_ids = keyphrase_ids[predicted_row_ids]

    tira_labels = keyphrase_ids == predicted_keyphrase_ids
    tira_probs = tira_matrix.max(axis=0)

    # monolingual ROC/AUC
    tira_fpr, tira_tpr, tira_thresholds = roc_curve(
        tira_labels,
        tira_probs,
    )
    tira_auc = roc_auc_score(tira_labels, tira_probs)

    drz_probs = drz_matrix.max(axis=0)
    drz_labels = np.zeros_like(drz_probs)

    # bilingual ROC/AUC
    all_probs = np.concatenate([tira_probs, drz_probs])
    all_labels = np.concatenate([tira_labels, drz_labels])

    all_fpr, all_tpr, all_thresholds = roc_curve(
        all_labels,
        all_probs,
    )
    all_auc = roc_auc_score(all_labels, all_probs)

    # english ROC/AUC
    tira_positive_mask = tira_labels.astype(bool)
    tira_positive_labels = tira_labels[tira_positive_mask]
    tira_positive_probs = tira_probs[tira_positive_mask]

    eng_probs = np.concatenate([tira_positive_probs, drz_probs])
    eng_labels = np.concatenate([tira_positive_labels, drz_labels])
    eng_fpr, eng_tpr, eng_thresholds = roc_curve(
        eng_labels,
        eng_probs,
    )
    eng_auc = roc_auc_score(eng_labels, eng_probs)


    return {
        'keyphrase': MAX_KEYWORD_STR,
        'tira>tira': {
            'fpr': tira_fpr.tolist(),
            'tpr': tira_tpr.tolist(),
            'auc': float(tira_auc),
        },
        'tira>english': {
            'fpr': eng_fpr.tolist(),
            'tpr': eng_tpr.tolist(),
            'auc': float(eng_auc),
        },
        'tira>tira/english': {
            'fpr': all_fpr.tolist(),
            'tpr': all_tpr.tolist(),
            'auc': float(all_auc),
        }
    }

if __name__ == '__main__':
    main()