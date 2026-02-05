#!/usr/bin/env python3
"""
Build list for evaluating keyphrase search.
Requires `text_preproc` to be run first!

This script constructs the following files:
- CER_MATRIX_PATH: character error rate matrix between keyphrases and all phrases
- KEYPHRASE_CSV: dataframe of keyphrases with difficulty statistics
- KEYPHRASE_LIST: JSON list of all keyphrases with positive/negative records
- CALIBRATION_LIST: JSON list of balanced subset for threshold calibration

```json
[
    {
        'keyphrase': $str,
        'keyphrase_idx': $int,
        'record_idcs': [$int, $int, ...]
        'easy': [$int, $int, ...]
        'medium': [$int, $int, ...]
        'hard': [$int, $int, ...]
    },
    ...
]
```

Where the 'easy', 'medium' and 'hard' keys map to lists of
record indices. This list maps all keyphrases to all positive
and negative records.
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
import os

# local imports
from src.dataloading import load_tira_asr, load_tira_drz
from src.constants import (
    PHRASES_CSV, KEYPHRASE_CSV, CER_MATRIX_PATH,
    CALIBRATION_LIST, KEYPHRASE_LIST, PHRASE2RECORDS_PATH,
    CALIBRATION_NUM_NEGATIVE, CALIBRATION_NUM_POSITIVE,
    ENGLISH_CALIBRATION_LIST,
)
from scripts.text_processing.get_words_and_phrases import build_merged_phrases_csv, build_phrases_csv, build_phrase2records


def define_keyphrases(unique_phrase_df, min_token_count=10):
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
    all_keyphrases = unique_phrase_df[keyphrase_mask]['phrase'].tolist()

    keyphrase_idcs = np.where(keyphrase_mask)[0].tolist()
    with open(output_path, 'w', encoding='utf8') as f:
        for idx in keyphrase_idcs:
            f.write(f"{idx}\n")

    return keyphrase_mask, all_keyphrases


def build_cer_matrix(
        all_phrases: List[str],
        keyphrase_mask: pd.Series,
        all_keyphrases: List[str],
        output_path: os.PathLike = CER_MATRIX_PATH,
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
                        keyphrase_mask, output_path: os.PathLike = KEYPHRASE_CSV):
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

        curr_keyphrase_mask = unique_phrase_df['phrase'] == keyphrase
        unique_phrase_df.loc[curr_keyphrase_mask, 'num_easy'] = easy_mask.sum()
        unique_phrase_df.loc[curr_keyphrase_mask, 'num_medium'] = medium_mask.sum()
        unique_phrase_df.loc[curr_keyphrase_mask, 'num_hard'] = hard_mask.sum()

    # Save keyphrases to CSV
    unique_phrase_df.to_csv(output_path, index_label='index')

    print(f"Saved keyphrase CSV to {output_path}")
    print(f"  Average easy negatives: {unique_phrase_df.loc[keyphrase_mask, 'num_easy'].mean():.1f}")
    print(f"  Average medium negatives: {unique_phrase_df.loc[keyphrase_mask, 'num_medium'].mean():.1f}")
    print(f"  Average hard negatives: {unique_phrase_df.loc[keyphrase_mask, 'num_hard'].mean():.1f}")

    # Also save full phrases CSV
    phrases_csv_path = PHRASES_CSV
    unique_phrase_df.to_csv(phrases_csv_path, index_label='index')
    print(f"Saved phrases CSV to {phrases_csv_path}")

    return unique_phrase_df


def build_keyphrase_lists(
    unique_phrase_df: pd.DataFrame,
    cer_matrix: np.ndarray,
    all_keyphrases: List[str],
    record2phrase: np.ndarray,
    keyphrase_list_path: Path,
    calibration_list_path: Path,
    calibration_num_negative: int = CALIBRATION_NUM_NEGATIVE,
    calibration_num_positive: int = CALIBRATION_NUM_POSITIVE,
    random_seed: int = 1337
):
    """
    Build KEYPHRASE_LIST and CALIBRATION_LIST: JSON files with
    positive/negative records.

    Structure:
    ```json
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
    ```

    CALIBRATION_LIST also contains an object with keyphrase "$eng"
    mapping to random negative records from the DRZ dataset.
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
        keyphrase = row['phrase']
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

def build_english_calibration_list(
    num_negative: int = CALIBRATION_NUM_NEGATIVE * 9,
    output_path: Path = ENGLISH_CALIBRATION_LIST,
) -> List[int]:
    """
    Build a calibration list entry for the English keyword.

    This entry contains only negative samples randomly selected
    from the DRZ dataset.

    Number is based on 9x the standard calibration negative count,
    such that when aggregated with the Tira keyphrases, the total
    number of English negatives is greater than for Tira, given that
    in real audio the majority of speech is expected to be non-target
    language.
    """
    print("\nBuilding English calibration list...")

    # Randomly select negative records from DRZ dataset
    drz_ds = load_tira_drz()
    num_drz_rows = len(drz_ds)
    drz_rows = np.arange(num_drz_rows).tolist()
    random_drz_negatives = random.sample(drz_rows, num_negative)

    with open(output_path, 'w', encoding='utf8') as f:
        for idx in random_drz_negatives:
            f.write(f"{idx}\n")

    return random_drz_negatives

def main():
    args = get_args()

    # Load files generated by `text_preproc.py`

    # TODO: `RECORD2PHRASE_PATH` is a CSV, not a text file!
    # load with pandas
    with open(PHRASE2RECORDS_PATH) as f:
        record2phrase = f.readlines()
    record2phrase = [int(line) for line in record2phrase]
    record2phrase = np.array(record2phrase)
    
    unique_phrase_df = pd.read_csv(PHRASES_CSV)
    all_phrases = unique_phrase_df['phrase'].tolist()

    # Build all files
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

    build_keyphrase_lists(
        unique_phrase_df=unique_phrase_df,
        cer_matrix=cer_matrix,
        all_keyphrases=all_keyphrases,
        record2phrase=record2phrase,
        keyphrase_list_path=KEYPHRASE_LIST,
        calibration_list_path=CALIBRATION_LIST,
        calibration_num_negative=args.calibration_num_negative,
        calibration_num_positive=args.calibration_num_positive,
        random_seed=args.random_seed,
    )

    drz_ds = load_tira_drz()
    num_drz_rows = len(drz_ds)
    print(f"\nLoaded Tira DRZ dataset with {num_drz_rows} records")
    build_english_calibration_list(
        num_negative=args.calibration_num_negative * 9,
        output_path=ENGLISH_CALIBRATION_LIST,
    )

    print("\n✓ All files built successfully!")


def get_args() -> Namespace:
    parser = ArgumentParser(description='Build KWS lists and associated files')
    parser.add_argument(
        '--min_token_count', type=int, default=10,
        help='Minimum token count for a phrase to be considered a keyphrase'
    )
    parser.add_argument(
        '--calibration_num_negative',
        type=int,
        default=CALIBRATION_NUM_NEGATIVE,
        help='Number of negative samples per difficulty level for calibration'
    )
    parser.add_argument(
        '--calibration_num_positive',
        type=int,
        default=CALIBRATION_NUM_POSITIVE,
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

