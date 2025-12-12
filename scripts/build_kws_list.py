#!/usr/bin/env python3
"""
Build KWS (Keyword Spotting) lists and associated files.

This script constructs the following files:
- MERGED_PHRASES_CSV: mapping of EAF text to FST-normalized text
- PHRASES_CSV: dataframe of unique phrases with statistics
- CER_MATRIX_PATH: character error rate matrix between keyphrases and all phrases
- PHRASE_PATH: text file with all unique phrases
- KEYPHRASE_CSV: dataframe of keyphrases with difficulty statistics
- RECORD2PHRASE_PATH: mapping of record indices to phrase indices
- KEYPHRASE_LIST: JSON list of all keyphrases with positive/negative records
- CALIBRATION_LIST: JSON list of balanced subset for threshold calibration
"""

import pandas as pd
from tqdm import tqdm
import random
import json
import numpy as np
from jiwer import cer
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List

# local imports
from dataloading import load_tira_asr
from constants import (
    PHRASE_PATH, KEYPHRASE_PATH, MERGED_PHRASES_CSV,
    PHRASES_CSV, KEYPHRASE_CSV, CER_MATRIX_PATH,
    CALIBRATION_LIST, KEYPHRASE_LIST, RECORD2PHRASE_PATH,
    LABELS_DIR
)


def build_merged_phrases_csv(df, output_path):
    """
    Build MERGED_PHRASES_CSV: mapping EAF text to FST-normalized text.

    Finds instances where FST normalization caused several dissimilar
    hand-transcribed sentences to merge.
    """
    print("Building merged phrases CSV...")

    # Build mapping from FST text to EAF text variants
    fst_to_eaf = {}
    fst_unique = df['fst_text'].unique().tolist()
    eaf_strs_encountered = set()

    for fst_text in tqdm(fst_unique, desc="Processing FST texts"):
        mask = df['fst_text'] == fst_text
        eaf_text = df.loc[mask, 'eaf_text'].unique().tolist()
        fst_to_eaf[fst_text] = eaf_text
        # ensure only one FST str per EAF str
        assert not any(eaf_str in eaf_strs_encountered for eaf_str in eaf_text)
        eaf_strs_encountered.update(eaf_text)

    # Create dataframe with unique EAF texts
    eaf_unique_df = df.drop_duplicates(subset=['eaf_text'])
    eaf_unique_df = eaf_unique_df.reset_index(drop=True)

    eaf_unique_df['num_eaf_variants'] = eaf_unique_df['fst_text']\
        .apply(fst_to_eaf.get)\
        .apply(len)
    eaf_unique_df = eaf_unique_df.sort_values('num_eaf_variants', ascending=False)

    # Save to CSV
    eaf_unique_df.to_csv(output_path, index_label='index')
    print(f"Saved merged phrases CSV to {output_path}")
    print(f"  Total unique FST texts: {len(fst_unique)}")
    print(f"  Total unique EAF texts: {len(eaf_unique_df)}")

    return eaf_unique_df


def build_phrases_csv(eaf_unique_df, df, output_path):
    """
    Build PHRASES_CSV: dataframe with unique FST strings and token counts.
    """
    print("\nBuilding phrases CSV...")

    # Create dataframe with unique FST phrases
    unique_phrase_df = eaf_unique_df.drop(columns='eaf_text')
    unique_phrase_df = unique_phrase_df.drop_duplicates(subset='fst_text')
    unique_phrase_df = unique_phrase_df.rename(columns={'fst_text':'keyphrase'})

    # Count occurrences of each phrase
    token_counts = df['fst_text'].value_counts()

    # Map token counts to phrases
    unique_phrase_df = unique_phrase_df.set_index('keyphrase')
    unique_phrase_df['token_count'] = token_counts
    unique_phrase_df = unique_phrase_df.reset_index()

    print(f"  Total unique phrases: {len(unique_phrase_df)}")
    print(f"  Token count statistics:\n{unique_phrase_df['token_count'].describe()}")

    unique_phrase_df.to_csv(output_path, index_label='index')
    print(f"Saved phrases CSV to {output_path}")

    return unique_phrase_df


def build_phrase_list(unique_phrase_df, output_path):
    """
    Build PHRASE_PATH: text file with all unique phrases (one per line).
    """
    print("\nBuilding phrase list...")

    all_phrases = unique_phrase_df['keyphrase'].tolist()

    with open(output_path, 'w', encoding='utf8') as f:
        for phrase in all_phrases:
            f.write(phrase + '\n')

    print(f"Saved phrase list to {output_path}")
    print(f"  Total phrases: {len(all_phrases)}")

    return all_phrases


def define_keyphrases(unique_phrase_df, min_token_count=10, output_path: str = KEYPHRASE_PATH):
    """
    Define keyphrases based on minimum token count threshold.

    Args:
        unique_phrase_df: DataFrame with unique phrases and token counts
        min_token_count: Minimum number of occurrences for a phrase to be a keyphrase
        output_path: Path to save keyphrase indices

    Returns:
        keyphrase_mask: Boolean mask indicating which phrases are keyphrases
        all_keyphrases: List of keyphrase strings
    """
    keyphrase_mask = unique_phrase_df['token_count'] >= min_token_count
    all_keyphrases = unique_phrase_df[keyphrase_mask]['keyphrase'].tolist()

    keyphrase_idcs = np.where(keyphrase_mask)[0].tolist()
    with open(output_path, 'w', encoding='utf8') as f:
        for idx in keyphrase_idcs:
            f.write(f"{idx}\n")

    return keyphrase_mask, all_keyphrases


def build_cer_matrix(
        all_phrases: List[str],
        keyphrase_mask: pd.Series,
        all_keyphrases: List[str],
        output_path: str,
        min_token_count=10
):
    """
    Build CER_MATRIX_PATH: matrix of character error rates between keyphrases and all phrases.
    """
    print("\nBuilding CER matrix...")

    print(f"  Keyphrases (>={min_token_count} occurrences): {len(all_keyphrases)}")
    print(f"  Total phrases: {len(all_phrases)}")
    print(f"  Matrix shape: ({len(all_keyphrases)}, {len(all_phrases)})")

    # Compute CER matrix
    cer_matrix = np.zeros((len(all_keyphrases), len(all_phrases)), dtype=float)

    for i, phrase1 in tqdm(enumerate(all_keyphrases),
                           total=len(all_keyphrases),
                           desc="Computing CER"):
        for j, phrase2 in enumerate(all_phrases):
            dist = cer(phrase1, phrase2)
            cer_matrix[i, j] = dist

    # Save matrix
    np.save(output_path, cer_matrix)
    print(f"Saved CER matrix to {output_path}")

    return cer_matrix, all_keyphrases, keyphrase_mask


def build_keyphrase_csv(unique_phrase_df, cer_matrix, all_keyphrases,
                        keyphrase_mask, output_path):
    """
    Build KEYPHRASE_CSV: dataframe of keyphrases with difficulty statistics.

    For each keyphrase, count negative examples by difficulty:
    - easy: CER > 0.67
    - medium: 0.33 < CER <= 0.67
    - hard: 0 < CER <= 0.33
    """
    print("\nBuilding keyphrase CSV...")

    unique_phrase_df['is_keyphrase'] = keyphrase_mask
    unique_phrase_df['num_easy'] = 0
    unique_phrase_df['num_medium'] = 0
    unique_phrase_df['num_hard'] = 0

    for keyphrase_idx, keyphrase in tqdm(enumerate(all_keyphrases),
                                         total=len(all_keyphrases),
                                         desc="Computing difficulty stats"):
        dists_to_keyphrase = cer_matrix[keyphrase_idx, :]

        easy_mask = dists_to_keyphrase > 0.67
        medium_mask = (dists_to_keyphrase <= 0.67) & (dists_to_keyphrase > 0.33)
        hard_mask = (dists_to_keyphrase > 0) & (dists_to_keyphrase <= 0.33)

        curr_keyphrase_mask = unique_phrase_df['keyphrase'] == keyphrase
        unique_phrase_df.loc[curr_keyphrase_mask, 'num_easy'] = easy_mask.sum()
        unique_phrase_df.loc[curr_keyphrase_mask, 'num_medium'] = medium_mask.sum()
        unique_phrase_df.loc[curr_keyphrase_mask, 'num_hard'] = hard_mask.sum()

    # Save keyphrases to CSV
    keyphrase_df = unique_phrase_df[keyphrase_mask].copy()
    keyphrase_df.to_csv(output_path, index_label='index')

    print(f"Saved keyphrase CSV to {output_path}")
    print(f"  Average easy negatives: {keyphrase_df['num_easy'].mean():.1f}")
    print(f"  Average medium negatives: {keyphrase_df['num_medium'].mean():.1f}")
    print(f"  Average hard negatives: {keyphrase_df['num_hard'].mean():.1f}")

    # Also save full phrases CSV
    phrases_csv_path = PHRASES_CSV
    unique_phrase_df.to_csv(phrases_csv_path, index_label='index')
    print(f"Saved phrases CSV to {phrases_csv_path}")

    return unique_phrase_df


def build_record2phrase(df, all_phrases, output_path):
    """
    Build RECORD2PHRASE_PATH: mapping of record index to phrase index.
    """
    print("\nBuilding record2phrase mapping...")

    # Create mapping from phrase to index
    phrase_to_idx = {phrase: idx for idx, phrase in enumerate(all_phrases)}

    # Map each record to its phrase index
    record2phrase = df['fst_text'].apply(phrase_to_idx.get).tolist()

    # Save to file
    with open(output_path, 'w', encoding='utf8') as f:
        for phrase_idx in record2phrase:
            f.write(f"{phrase_idx}\n")

    print(f"Saved record2phrase mapping to {output_path}")
    print(f"  Total records: {len(record2phrase)}")

    return np.array(record2phrase)


def build_keyphrase_lists(
    unique_phrase_df: pd.DataFrame,
    cer_matrix: np.ndarray,
    all_keyphrases: List[str],
    record2phrase: np.ndarray,
    keyphrase_list_path: Path,
    calibration_list_path: Path,
    calibration_num_negative: int = 50,
    calibration_num_positive: int = 10,
    random_seed: int = 1337
):
    """
    Build KEYPHRASE_LIST and CALIBRATION_LIST: JSON files with
    positive/negative records.

    Structure:
    [
        {
            'keyphrase': str,
            'keyphrase_idx': int,
            'record_idcs': [int, ...],  # positive records
            'easy': {
                'phrase_idcs': [int, ...],
                'record_idcs': [int, ...]
            },
            'medium': {...},
            'hard': {...}
        },
        ...
    ]
    """
    print("\nBuilding keyphrase lists...")

    random.seed(random_seed)

    # Determine which keyphrases have enough negatives for calibration
    keyphrase_mask = unique_phrase_df['is_keyphrase']
    has_easy = unique_phrase_df['num_easy'] >= calibration_num_negative
    has_medium = unique_phrase_df['num_medium'] >= calibration_num_negative
    has_hard = unique_phrase_df['num_hard'] >= calibration_num_negative
    has_negative = has_easy & has_medium & has_hard
    unique_phrase_df['in_calibration_set'] = has_negative

    print(
        "  Keyphrases in calibration set: "+\
        f"{has_negative.sum()}/{keyphrase_mask.sum()}"
    )

    # Build lists
    keyphrase_list = []
    calibration_list = []

    for _, row in tqdm(
        unique_phrase_df[unique_phrase_df['is_keyphrase']].iterrows(),
        total=len(all_keyphrases),
        desc="Building lists",
    ):
        # Get positive records (all records with this keyphrase)
        row_i = row.name
        keyphrase = row['keyphrase']
        keyphrase_i = all_keyphrases.index(keyphrase)
        positive_mask = record2phrase == row_i
        positive_record_idcs = np.where(positive_mask)[0].tolist()

        # Build negative sets by difficulty
        dists_to_keyphrase = cer_matrix[keyphrase_i, :]

        easy_mask = dists_to_keyphrase > 0.67
        medium_mask = (dists_to_keyphrase <= 0.67) & (dists_to_keyphrase > 0.33)
        hard_mask = (dists_to_keyphrase > 0) & (dists_to_keyphrase <= 0.33)

        # Create keyphrase object for full list
        keyphrase_obj = {
            'keyphrase': keyphrase,
            'keyphrase_idx': row_i,
            'record_idcs': positive_record_idcs,
            'easy': [],
            'medium': [],
            'hard': [],
        }

        # Add negative examples for each difficulty
        for mask, difficulty in [(easy_mask, 'easy'), (medium_mask, 'medium'), (hard_mask, 'hard')]:
            negative_phrase_idcs = np.where(mask)[0].tolist()
            negative_record_mask = np.isin(record2phrase, negative_phrase_idcs)
            negative_record_idcs = np.where(negative_record_mask)[0].tolist()
            keyphrase_obj[difficulty] = negative_record_idcs

        keyphrase_list.append(keyphrase_obj)

        # Build calibration list (subset with balanced samples)
        if row['in_calibration_set']:
            calibration_positive_idcs = random.sample(positive_record_idcs, calibration_num_positive)
            calibration_obj = {
                'keyphrase': keyphrase,
                'keyphrase_idx': row_i,
                'record_idcs': calibration_positive_idcs,
                'easy': [],
                'medium': [],
                'hard': [],
            }

            # Sample negative records for calibration
            for difficulty in ['easy', 'medium', 'hard']:
                all_negative_record_idcs = keyphrase_obj[difficulty]
                all_negative_phrase_idcs = np.unique(record2phrase[all_negative_record_idcs]).tolist()

                # Sample phrases first, then get their records
                sampled_phrase_idcs = random.sample(
                    all_negative_phrase_idcs, calibration_num_negative
                )
                sampled_record_idcs = []
                for phrase_idx in sampled_phrase_idcs:
                    phrase_records = np.where(record2phrase == phrase_idx)[0].tolist()
                    # Take one random record per phrase if multiple exist
                    sampled_record_idcs.append(random.choice(phrase_records))

                calibration_obj[difficulty] = sampled_record_idcs

            calibration_list.append(calibration_obj)

    # Save JSON files
    with open(keyphrase_list_path, 'w', encoding='utf8') as f:
        json.dump(keyphrase_list, f, indent=2, ensure_ascii=False)
    print(f"Saved keyphrase list to {keyphrase_list_path}")
    print(f"  Total keyphrases: {len(keyphrase_list)}")

    with open(calibration_list_path, 'w', encoding='utf8') as f:
        json.dump(calibration_list, f, indent=2, ensure_ascii=False)
    print(f"Saved calibration list to {calibration_list_path}")
    print(f"  Total keyphrases: {len(calibration_list)}")


def main():
    args = get_parser()

    # Ensure output directory exists
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Tira ASR dataset...")
    ds = load_tira_asr()
    print(f"Loaded dataset with {len(ds)} records")

    # Prepare dataframe
    print("\nPreparing dataframe...")
    colmap = {'transcription': 'eaf_text', 'rewritten_transcript': 'fst_text'}
    cols_to_drop = set(ds.column_names) - set(colmap.keys())
    ds_noaudio = ds.remove_columns(list(cols_to_drop))
    df = ds_noaudio.to_pandas()
    df = df.rename(columns=colmap)
    print(f"DataFrame shape: {df.shape}")

    # Build all files
    eaf_unique_df = build_merged_phrases_csv(df, MERGED_PHRASES_CSV)

    unique_phrase_df = build_phrases_csv(eaf_unique_df, df, PHRASES_CSV)

    all_phrases = build_phrase_list(unique_phrase_df, PHRASE_PATH)
    keyphrase_mask, all_keyphrases = define_keyphrases(unique_phrase_df, args.min_token_count)

    cer_matrix, all_keyphrases, keyphrase_mask = build_cer_matrix(
        keyphrase_mask=keyphrase_mask,
        all_keyphrases=all_keyphrases,
        all_phrases=all_phrases,
        output_path=CER_MATRIX_PATH,
        min_token_count=args.min_token_count
    )

    unique_phrase_df = build_keyphrase_csv(
        unique_phrase_df, cer_matrix, all_keyphrases,
        keyphrase_mask, KEYPHRASE_CSV
    )

    record2phrase = build_record2phrase(df, all_phrases, RECORD2PHRASE_PATH)

    build_keyphrase_lists(
        unique_phrase_df=unique_phrase_df,
        cer_matrix=cer_matrix,
        all_keyphrases=all_keyphrases,
        record2phrase=record2phrase,
        keyphrase_list_path=KEYPHRASE_LIST,
        calibration_list_path=CALIBRATION_LIST,
        calibration_num_negative=args.calibration_num_negative,
        calibration_num_positive=args.calibration_num_positive,
        random_seed=args.random_seed
    )

    print("\n✓ All files built successfully!")


def get_parser() -> Namespace:
    parser = ArgumentParser(description='Build KWS lists and associated files')
    parser.add_argument(
        '--min_token_count', type=int, default=10,
        help='Minimum token count for a phrase to be considered a keyphrase'
    )
    parser.add_argument(
        '--calibration_num_negative', type=int, default=50,
        help='Number of negative samples per difficulty level for calibration'
    )
    parser.add_argument(
        '--calibration_num_positive', type=int, default=10,
        help='Number of positive samples for calibration'
    )
    parser.add_argument(
        '--random_seed', type=int, default=1337,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

