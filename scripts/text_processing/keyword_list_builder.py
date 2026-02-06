#!/usr/bin/env python3
"""
Build list for evaluating keyword search.
Requires `text_preproc` to be run first!

This script constructs the KEYWORD_LIST JSON file, which contains
information about keywords used for the Interspeech 2026 KWS experiment.
See `build_keyword_list` for JSON data structure.
"""

import pandas as pd
import random
import json
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Union
from unidecode import unidecode

# local imports
from src.constants import (
    WORDS_CSV, WORD2PHRASE_PATH, KEYWORD_LIST,
    KEYWORD_SENTENCES, KEYWORDS_DIR, KEYWORDS_CSV,
    PHRASE2RECORDS_PATH, PHRASES_CSV
)

ASYLLABIC_VERB_ROOTS = ['p', 't̪(1)', 't̪(2)', 'n']

def build_keyword_list(
        word_df: pd.DataFrame,
        word2phrase: pd.DataFrame,
        phrase2records: pd.DataFrame,
        phrase_count: int=10,
        num_keywords: int=30,
        min_keyword_length: int=5,
        random_seed: int=1337,
    ) -> Tuple[List[Dict[str, Union[str, List[int]]]], pd.DataFrame]:
    """
    Samples a set of keywords from the word DataFrame with specified token
    count and number of keywords, and maps keywords to audio records from
    the Tira ASR dataset. Returns a list of keyword dicts with the following
    structure:
    ```json
    [
        {
            'keyword': $str,
            'keyword_idcs': [$int, $int, ... ], # added by `make_keyword_ds.py`
            'positive_phrase_idcs': [$int, $int, ...],
            'positive_record_idcs': [$int, $int, ...],
        },
        ...
    ]
    ```
    - The 'keyword' key gives the string of the keyword

    - The 'positive_phrase_idcs' key maps to the indices of phrases that contain the
        keyword (i.e., positive examples)
    - The 'positive_record_idcs' key maps to the indices of audio records that
        contain the keyword (one for each phrase in 'positive_phrase_idcs')
    - The key 'keyword_idcs' maps to the indices of all query tokens for the given
        keyword in the keyword dataset (built using `make_keyword_ds.py`). This key
        is NOT built in this function but is added later.
    

    Args:
        word_df: DataFrame with unique phrases and token counts
        word2phrase: Mapping from words to phrases containing the word
        phrase2records: Mapping from phrases to records containing the phrase
        phrase_count: Number of phrases for each keyword.
        num_keywords: Number of keywords to select.
        min_keyword_length: Minimum length of each keyword in characters.
        random_seed: Random seed for reproducibility.
    Returns:
        keyword_list: List of keyword dicts
        keywords: pd.DataFrame containing only the keywords selected for the experiment
    """
    phrase_mask = word_df['phrase_count'] >= phrase_count
    print(f" Words with >= {phrase_count} phrases: {phrase_mask.sum()}")

    # apply unidecode so that combining diacritics don't affect length
    unidecode_words = word_df['word'].apply(unidecode)
    # ŋ character is two chars in unidecode, so we need to account for that
    unidecode_words = unidecode_words.str.replace('ng', 'ŋ')

    length_mask = unidecode_words.str.len() >= min_keyword_length
    print(f" Words with length >= {min_keyword_length}: {length_mask.sum()}")

    # exclude asyllabic verb roots
    asyllabic_mask = ~word_df['lemma'].isin(ASYLLABIC_VERB_ROOTS)
    print(f" Words excluding asyllabic verb roots: {asyllabic_mask.sum()}")

    candidate_mask = phrase_mask & length_mask & asyllabic_mask
    print(f" Candidate words: {candidate_mask.sum()}")
    print(f" Sampling {num_keywords} keywords...")

    keyword_indices = []
    keyword_list = []
    i = 0

    while len(keyword_indices) < num_keywords:
        if candidate_mask.sum() == 0:
            raise ValueError(
                f"Not enough candidate words to sample {num_keywords} keywords "
                f"with phrase_count={phrase_count} and min_keyword_length={min_keyword_length}. "
                f"Consider lowering these thresholds."
            )


        # sample keyword row
        sampled_keyword_index = random.Random(random_seed+i).choice(
            word_df[candidate_mask].index.tolist()
        )
        word = word_df.loc[sampled_keyword_index, 'word']
        
        phrase_idcs = word2phrase[
            word2phrase['word_idx'] == sampled_keyword_index
        ]['phrase_idx'].tolist()

        # filter out phrases that contain any already selected keywords
        # (to ensure negative examples are clean)
        for idx in keyword_indices:
            existing_phrases = word2phrase[
                word2phrase['word_idx'] == idx
            ]['phrase_idx'].tolist()
            phrase_idcs = [idx for idx in phrase_idcs if idx not in existing_phrases]

        # sanity check to make sure we have enough phrases
        if len(phrase_idcs) < phrase_count:
            print(f"Insufficient phrases for word '{word}' (only {len(phrase_idcs)} available after filtering), skipping...")
            candidate_mask[sampled_keyword_index] = False
            continue

        # sample phrase_count phrases randomly
        random.Random(random_seed).shuffle(phrase_idcs)
        positive_phrase_idcs = phrase_idcs[:phrase_count]

        # get one positive record for each phrase
        positive_record_idcs = []
        for phrase_idx in positive_phrase_idcs:
            records = phrase2records[phrase2records['phrase_idx'] == phrase_idx]['record_idx'].tolist()
            if len(records) == 0:
                raise ValueError(
                    f"No records found for phrase index {phrase_idx} "
                    f"containing word '{word}'"
                )
            record_idx = random.Random(random_seed).choice(records)
            positive_record_idcs.append(record_idx)

        # block out all words with the same lemma
        lemma = word_df.loc[sampled_keyword_index, 'lemma']
        lemma_mask = word_df['lemma'] == lemma
        candidate_mask = candidate_mask & ~lemma_mask

        # append index and increment i
        keyword_indices.append(sampled_keyword_index)
        i+=1

        keyword_list.append({
            'keyword': word,
            'positive_phrase_idcs': positive_phrase_idcs,
            'positive_record_idcs': positive_record_idcs,
        })

        print(f" Selected keyword '{word}'")

    keyword_df = word_df.loc[keyword_indices]
    keywords = keyword_df['word'].tolist()

    print(" Sampled keywords:")
    print("\n".join(keywords))

    return keyword_list, keyword_df

def get_all_phrase_idcs(
        keywords: List[str],
        keyword_list: List[Dict[str, Union[str, List[int]]]],
        phrase_list: List[str],
        phrase2records: pd.DataFrame,
        negative_phrase_count: int=700,
        random_seed: int=1337,
    ) -> pd.DataFrame:
    """
    Get all positive and negative phrase indices for the given keywords.
    Positive indices are specified by `keyword_list`, negative indices are
    sampled randomly from phrases that do not contain any of the keywords.

    args:
        keywords: List of keyword strings
        keyword_list: List of keyword dicts
        phrase_list: List of all phrases
        phrase2records: pd.DataFrame mapping phrases to records
        negative_phrase_count: Number of negative phrases to sample
        random_seed: Random seed for reproducibility
    returns:
        positive and negative phrase indices
    """

    positive_phrase_idcs = []
    positive_record_idcs = []
    for keyword_dict in keyword_list:
        positive_phrase_idcs.extend(keyword_dict['positive_phrase_idcs'])
        positive_record_idcs.extend(keyword_dict['positive_record_idcs'])

    positive_phrase_idcs = set(positive_phrase_idcs)
    positive_record_idcs = set(positive_record_idcs)

    # sanity check: should have same number of positive phrases and records
    assert len(positive_phrase_idcs) == len(positive_record_idcs)

    print(f" Total unique positive phrases: {len(positive_phrase_idcs)}")


    print(" Collecting phrases that do not contain any keywords...")
    negative_phrase_idcs = []
    for idx, phrase in enumerate(phrase_list):
        if any(keyword in phrase for keyword in keywords):
            continue
        negative_phrase_idcs.append(idx)
    print(f" Negative candidate phrases: {len(negative_phrase_idcs)}")

    print(" Collecting records for negative phrases...")
    negative_record_idcs = []

    for idx in negative_phrase_idcs:
        record_idcs = phrase2records[phrase2records['phrase_idx'] == idx]['record_idx'].tolist()
        record_idx = random.Random(random_seed).choice(record_idcs)
        negative_record_idcs.append(record_idx)

    # sanity check: ensure no overlap between positive and negative indices
    assert not positive_record_idcs.intersection(negative_record_idcs)
    assert not positive_phrase_idcs.intersection(set(negative_phrase_idcs))
    
    print(f" Negative candidate records: {len(negative_record_idcs)}")
    sampled_negative_idcs = random.Random(random_seed).sample(
        negative_record_idcs,
        k=negative_phrase_count,
    )
    print(f" Sampled {len(sampled_negative_idcs)} negative records.")
    
    positive_data = pd.DataFrame({
        'phrase_idx': list(positive_phrase_idcs),
        'record_idx': list(positive_record_idcs),
        'is_positive': True,
    })
    negative_data = pd.DataFrame({
        'phrase_idx': negative_phrase_idcs,
        'record_idx': negative_record_idcs,
        'is_positive': False,
    })
    # filter out only sampled negative records
    negative_data = negative_data[negative_data['record_idx'].isin(sampled_negative_idcs)]

    all_data = pd.concat([positive_data, negative_data], ignore_index=True)
    print(f" Total test phrases (positive + negative): {len(all_data)}")
    return all_data

def main():
    args = get_args()

    # Load files generated by `text_preproc.py`
    word_df = pd.read_csv(WORDS_CSV)
    print(f"Loaded {len(word_df)} unique words from {WORDS_CSV}")

    word2phrase = pd.read_csv(WORD2PHRASE_PATH)
    print(f"Loaded word2phrase mapping from {WORD2PHRASE_PATH}")

    phrase2records = pd.read_csv(PHRASE2RECORDS_PATH)
    print(f"Loaded phrase2records mapping from {PHRASE2RECORDS_PATH}")

    print(f"\nBuilding keyword list with phrase_count={args.phrase_count} and num_keywords={args.num_keywords}...")
    KEYWORDS_DIR.mkdir(parents=True, exist_ok=True)

    # Build keyword list
    keyword_list, keyword_df = build_keyword_list(
        word_df,
        word2phrase,
        phrase2records,
        phrase_count=args.phrase_count,
        num_keywords=args.num_keywords,
        min_keyword_length=args.min_keyword_length,
        random_seed=args.random_seed,
    )
    keywords = keyword_df['word'].tolist()

    print(f"\nSaving keywords to {KEYWORDS_CSV}...")
    keyword_df.to_csv(KEYWORDS_CSV, index_label='word_idx')

    # Get test phrase indices (positive + negative)
    unique_phrase_df = pd.read_csv(PHRASES_CSV)

    all_phrase_data = get_all_phrase_idcs(
        keywords,
        keyword_list,
        phrase_list=unique_phrase_df['phrase'].tolist(),
        phrase2records=phrase2records,
        negative_phrase_count=args.negative_phrase_count,
        random_seed=args.random_seed,
    )

    print(f"Saving keyword list with {len(keyword_list)} keywords to {KEYWORD_LIST}...")
    with open(KEYWORD_LIST, 'w', encoding='utf8') as f:
        json.dump(keyword_list, f, indent=4, ensure_ascii=False)

    print(f"Saving keyword positive and negative phrase indices to {KEYWORD_SENTENCES}...")
    all_phrase_data.to_csv(KEYWORD_SENTENCES, index=False)

    print("\n✓ All keyword datafiles built successfully!")


def get_args() -> Namespace:
    parser = ArgumentParser(description='Build KWS lists and associated files')
    parser.add_argument(
        '--phrase_count', type=int, default=10,
        help='Number of tokens to be used for keywords.'
    )
    parser.add_argument(
        '--num_keywords', type=int, default=30,
        help='Number of keywords to generate.'
    )
    parser.add_argument(
        '--min_keyword_length', type=int, default=5,
        help='Minimum length of each keyword in characters.'
    )
    parser.add_argument(
        '--negative_phrase_count', type=int, default=700,
        help='Number of negative phrases that contain no keywords.'
    )
    parser.add_argument(
        '--random_seed', type=int, default=1337,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

