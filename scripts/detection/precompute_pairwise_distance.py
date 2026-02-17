"""
For a given Lhotse FeatureSet, compute the pairwise distance matrix between
all pairs of keyword / test phrase embeddings, then save as a list of matrices
in a .npz file under data/distance/$feature_name alongside a CSV file 'manifest.csv'
containing the  keyword and test phrase indices for each matrix.
"""

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from tira_kws.dataloading import load_kws_cuts
from tira_kws.distance import (
    pairwise_cosine_similarity, pad_arrays_or_tensors,
    pad_matrices
)
from tira_kws.constants import DEVICE, DISTANCE_DIR, KEYWORD_SENTENCES
import argparse
import re

def main():
    DISTANCE_DIR.mkdir(exist_ok=True)

    args = get_args()
    feature_name = args.feature_name
    feature_dir = DISTANCE_DIR / feature_name
    distance_matrix_dir = feature_dir / "distance_matrices"

    feature_dir.mkdir(exist_ok=True, parents=True)
    distance_matrix_dir.mkdir(exist_ok=True, parents=True)

    # Load keyword cuts
    print(f"Loading cuts for feature set '{feature_name}'...")
    keyword_cuts, test_phrase_cuts = load_kws_cuts(feature_name)
    keyword_cuts = keyword_cuts.to_eager()
    test_phrase_cuts = test_phrase_cuts.to_eager()

    # Compute pairwise distance matrix
    # iter through batches of keyword and test phrase cuts
    # and compute pairwise cosine similarity for each batch
    # then save results to a list of matrices
    # also keep track of the corresponding keyword and test phrase indices for each matrix
    manifest = []

    keyword_batch_range = list(range(0, len(keyword_cuts), args.batch_size))
    for i in tqdm(keyword_batch_range, desc="Computing pairwise distance matrices"):
        batch_matrices = []
        keyword_batch = keyword_cuts[i:i+args.batch_size]

        keyword_features = [cut.load_features() for cut in keyword_batch]
        keyword_features = [torch.Tensor(feature) for feature in keyword_features]
        
        keyword_lengths = [feature.shape[0] for feature in keyword_features]
        
        positive_record_ids = [cut.id for cut in keyword_batch]
        id_stem_regex = r'(.+)-\d+' # matches the stem of the cut ID before the final dash 
        # and number, which corresponds to the alignment index for the keyword in the sentence
        positive_record_ids = [re.match(id_stem_regex, cut_id).group(1) for cut_id in positive_record_ids]

        keyword_features = pad_arrays_or_tensors(keyword_features).to(DEVICE)

        test_phrase_batch_range = range(0, len(test_phrase_cuts), args.batch_size)
        for j in tqdm(test_phrase_batch_range):
            test_phrase_batch = test_phrase_cuts[j:j+args.batch_size]
            
            test_phrase_features = [cut.load_features() for cut in test_phrase_batch]
            test_phrase_features = [torch.Tensor(feature) for feature in test_phrase_features]

            test_phrase_lengths = [feature.shape[0] for feature in test_phrase_features]
            test_record_ids = [cut.id for cut in test_phrase_batch]

            test_phrase_features = pad_arrays_or_tensors(test_phrase_features).to(DEVICE)

            distance_matrix_4d = 1-pairwise_cosine_similarity(keyword_features, test_phrase_features)
            inner_batch_matrices = torch.flatten(distance_matrix_4d, start_dim=0, end_dim=1).cpu().numpy()

            batch_matrices.extend(inner_batch_matrices)

            # Update manifest with keyword and test phrase indices
            for positive_record_id, keyword_length in zip(positive_record_ids, keyword_lengths):
                for test_record_id, test_phrase_length in zip(test_record_ids, test_phrase_lengths):
                    manifest.append({
                        'positive_record_id': positive_record_id,
                        'test_record_id': test_record_id,
                        'keyword_length': keyword_length,
                        'test_phrase_length': test_phrase_length,
                    })

        padded_batch = pad_matrices(batch_matrices)
        batch_start = i*len(test_phrase_cuts)
        batch_end = batch_start + len(batch_matrices)
        distance_matrix_path = distance_matrix_dir / f"distance_matrices_{batch_start}_{batch_end}.npz"
        np.savez_compressed(distance_matrix_path, batch_matrices=padded_batch)
    
    print(f"Saved distance matrices to {distance_matrix_dir}")

    print("Saving manifest...")
    manifest_path = feature_dir / "manifest.csv"
    manifest_df = pd.DataFrame(manifest)

    # associate word indices with positive phrase IDs in manifest
    keyword_sentences = pd.read_csv(KEYWORD_SENTENCES)
    phrase_idx2word_idx = {
        int(row['phrase_idx']): int(row['word_idx'])
        for _, row in keyword_sentences.iterrows()
    }
    manifest_df['positive_record_id'] = manifest_df['positive_record_id'].astype(int)
    manifest_df['keyword_id'] = manifest_df['positive_record_id'].map(phrase_idx2word_idx)
    assert manifest_df['keyword_id'].isna().sum() == 0,\
        "Expected all positive record IDs in manifest to have a corresponding keyword ID, but got some missing values"
    print("Associated keyword IDs with positive phrase IDs in manifest")

    print(manifest_df.head())

    manifest_df.to_csv(manifest_path, index_label="matrix_index")
    print(f"Saved manifest to {manifest_path}")

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