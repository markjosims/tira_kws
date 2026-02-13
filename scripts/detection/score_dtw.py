"""
Load a pairwise distance matrix and manifest file computed by
compute_pairwise_distance.py and compute the DTW score for each
pair of keyword / test phrase, then update the manifest file with
the DTW score and save as a new CSV file 'manifest_dtw.csv' under
data/distance/$feature_name.
"""

from src.constants import DISTANCE_DIR
import argparse
from glob import glob
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from src.dtw import batched_subseq_dtw

def main():
    args = get_args()
    feature_name = args.feature_name

    feature_distance_dir = DISTANCE_DIR / feature_name
    distance_matrices = feature_distance_dir.glob(f"distance_matrices_*.npz")
    distance_matrices = list(distance_matrices)

    manifest_path = feature_distance_dir / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    manifest_df['dtw_score'] = np.nan # initialize dtw_score column with NaN values
    
    # get the index of the dtw_score column for later assignment
    dtw_score_col_loc = manifest_df.columns.get_loc('dtw_score')

    # matches the start and end indices of the batch in the manifest
    matrix_index_regex = r'distance_matrices_(\d+)_(\d+)'

    for distance_matrix_path in tqdm(distance_matrices, desc="Computing DTW scores"):
        distance_matrix = np.load(distance_matrix_path)['batch_matrices']
        batch_start, batch_end = re.match(matrix_index_regex, distance_matrix_path.stem).groups()
        batch_start, batch_end = int(batch_start), int(batch_end)
        batch_manifest_df = manifest_df.iloc[batch_start:batch_end]

        dtw_scores = []
        batch_range = list(range(0, len(batch_manifest_df), args.batch_size))
        for inner_batch_start in tqdm(batch_range, desc="Computing DTW scores for batch"):
            inner_batch_end = min(inner_batch_start + args.batch_size, len(batch_manifest_df))

            keyword_lengths = batch_manifest_df.iloc[inner_batch_start:inner_batch_end]['keyword_length']
            test_phrase_lengths = batch_manifest_df.iloc[inner_batch_start:inner_batch_end]['test_phrase_length']

            keyword_lengths = keyword_lengths.to_numpy()
            test_phrase_lengths = test_phrase_lengths.to_numpy()

            distance_submatrix = distance_matrix[inner_batch_start:inner_batch_end]
            dtw_score = batched_subseq_dtw(distance_submatrix, keyword_lengths, test_phrase_lengths)
            dtw_scores.extend(dtw_score)

        manifest_df.iloc[batch_start:batch_end, dtw_score_col_loc] = dtw_scores

    output_path = feature_distance_dir / "manifest_dtw.csv"
    manifest_df.to_csv(output_path, index=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_name",
        type=str,
        default="xlsr",
        help="Name of the feature set to compute pairwise distances for.",
    )
    parser.add_argument(
        "--batch_size", '-b',
        type=int,
        default=128,
        help="Batch size for computing pairwise distances. Adjust based on available memory.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()