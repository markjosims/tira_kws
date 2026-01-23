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
    WORDS_CSV, WORD2PHRASE_PATH, KEYWORD_LIST, RECORD2PHRASE_PATH,
    KEYWORD_QUERY_RECORDS, KEYWORD_TESTPHRASE_RECORDS,
)

def build_keyword_list(
        word_df: pd.DataFrame,
        word2phrase: List[List[int]],
        token_count: int=10,
        num_keywords: int=30,
        min_keyword_length: int=5,
            random_seed: int=1337,
    ) -> Tuple[List[Dict[str, Union[str, List[int]]]], List[str]]:
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
            'keyword_record_idcs': [$int, $int, ...]
            'positive_record_idcs': [$int, $int, ...]
        },
        ...
    ]
    ```
    - The 'keyword' key gives the string of the keyword
    - The 'keyword_record_idcs' key maps to the indices of records that query
        recordings are sampled from.
    - The 'positive_record_idcs' key maps to the indices of records that contain the
        keyword (i.e., positive examples) used as targets (rather than queries)
        during KWS evaluation.
    - The key 'keyword_idcs' maps to the indices of all query tokens for the given
        keyword in the keyword dataset (built using `make_keyword_ds.py`). This key
        is NOT built in this function but is added later.
    

    Args:
        word_df: DataFrame with unique phrases and token counts
        word2phrase: Mapping from words to phrases containing the word
        token_count: Number of tokens for each keyword.
        num_keywords: Number of keywords to select.
        min_keyword_length: Minimum length of each keyword in characters.
        random_seed: Random seed for reproducibility.
    Returns:
        keyword_list: List of keyword dicts
        keywords: List of keyword strings
    """
    # multiply token count by 2 to account for sampling both
    # queries and positive targets
    token_mask = word_df['token_count'] >= token_count*2
    print(f" Words with >= {token_count*2} tokens: {token_mask.sum()}")

    # apply unidecode so that combining diacritics don't affect length
    unidecode_words = word_df['word'].apply(unidecode)
    length_mask = unidecode_words.str.len() >= min_keyword_length
    print(f" Words with length >= {min_keyword_length}: {length_mask.sum()}")

    candidate_mask = token_mask & length_mask
    print(f" Candidate words: {candidate_mask.sum()}")
    print(f" Sampling {num_keywords} keywords...")

    keyword_df = word_df[candidate_mask].sample(
        n=num_keywords,
        random_state=random_seed,
    )
    keyword_list = []

    for index, word in keyword_df['word'].items(): # type: ignore

        index: int
        phrase_idcs = word2phrase[index]

        # split phrases into query and positive sets
        random.Random(random_seed).shuffle(phrase_idcs)
        phrase_idcs = phrase_idcs[:token_count*2]
        split_idx = len(phrase_idcs) // 2
        query_phrase_idcs = phrase_idcs[:split_idx]
        positive_phrase_idcs = phrase_idcs[split_idx:]

        keyword_list.append({
            'keyword': word,
            'keyword_record_idcs': query_phrase_idcs,
            'positive_record_idcs': positive_phrase_idcs,
        })

    print(" Sampled keywords:")
    keywords = keyword_df['word'].tolist()
    print("\n".join(keywords))

    return keyword_list, keywords

def get_test_phrase_idcs(
        keywords: List[str],
        keyword_list: List[Dict[str, Union[str, List[int]]]],
        word_df: pd.DataFrame,
        record2phrase: List[List[int]],
        negative_phrase_count: int=700,
        random_seed: int=1337,
    ) -> List[int]:
    """
    Get all positive and negative phrase indices for the given keywords.
    Positive indices are specified by `keyword_list`, negative indices are
    sampled randomly from phrases that do not contain any of the keywords.

    args:
        keywords: List of keyword strings
        keyword_list: List of keyword dicts
        word_df: DataFrame with unique phrases and token counts
        record2phrase: Mapping from records to phrases contained in the record
        negative_phrase_count: Number of negative phrases to sample
        random_seed: Random seed for reproducibility
    returns:
        all_phrase_idcs: List of all positive and negative phrase indices
    """

    all_positive_idcs = []
    for keyword_dict in keyword_list:
        all_positive_idcs.extend(keyword_dict['positive_record_idcs'])
    all_positive_idcs = set(all_positive_idcs)

    no_keyword_mask = ~word_df['word'].isin(keywords)
    negative_candidate_df = word_df[no_keyword_mask]
    print(f" Negative candidate phrases: {len(negative_candidate_df)}")

    negative_phrase_idcs = negative_candidate_df.index.tolist()
    negative_record_idcs = []

    for idx in negative_phrase_idcs:
        record_idcs = record2phrase[idx]
        negative_record_idcs.extend(record_idcs)
    negative_record_idcs = set(negative_record_idcs)

    # sanity check: ensure no overlap between positive and negative indices
    assert not all_positive_idcs.intersection(negative_record_idcs)
    negative_record_idcs = list(negative_record_idcs)
    
    print(f" Negative candidate records: {len(negative_record_idcs)}")
    sampled_negative_idcs = random.sample(
        negative_record_idcs,
        k=negative_phrase_count,
    )
    print(f" Sampled {len(sampled_negative_idcs)} negative records.")
    test_phrase_idcs = list(all_positive_idcs) + sampled_negative_idcs
    return test_phrase_idcs

def get_query_phrase_idcs(
        keyword_list: List[Dict[str, Union[str, List[int]]]],
    ) -> List[int]:
    """
    Get all query phrase indices from the keyword list.
    Simply performs a union over all 'keyword_record_idcs' entries.
    args:
        keyword_list: List of keyword dicts
    returns:
        query_phrase_idcs: List of all query phrase indices
    """
    all_query_idcs = []
    for keyword_dict in keyword_list:
        all_query_idcs.extend(keyword_dict['keyword_record_idcs'])
    all_query_idcs = set(all_query_idcs)
    query_phrase_idcs = list(all_query_idcs)
    return query_phrase_idcs

def main():
    args = get_args()

    # Load files generated by `text_preproc.py`
    word_df = pd.read_csv(WORDS_CSV)
    print(f"Loaded {len(word_df)} unique words from {WORDS_CSV}")

    with open(WORD2PHRASE_PATH, 'r', encoding='utf8') as f:
        word2phrase = [list(map(int, line.strip().split())) for line in f.readlines()]
    print(f"Loaded word2phrase mapping from {WORD2PHRASE_PATH}")

    with open(RECORD2PHRASE_PATH, 'r', encoding='utf8') as f:
        record2phrase = [list(map(int, line.strip().split())) for line in f.readlines()]
    print(f"Loaded record2phrase mapping from {RECORD2PHRASE_PATH}")

    print(f"Building keyword list with token_count={args.token_count} and num_keywords={args.num_keywords}...")

    # Build keyword list
    keyword_list, keywords = build_keyword_list(
        word_df,
        word2phrase,
        token_count=args.token_count,
        num_keywords=args.num_keywords,
        min_keyword_length=args.min_keyword_length,
        random_seed=args.random_seed,
    )

    # Get test phrase indices (positive + negative)
    test_phrase_idcs = get_test_phrase_idcs(
        keywords,
        keyword_list,
        word_df,
        record2phrase,
        negative_phrase_count=args.negative_phrase_count,
        random_seed=args.random_seed,
    )
    print(f" Total test phrases (positive + negative): {len(test_phrase_idcs)}")

    # Get query phrase indices
    query_phrase_idcs = get_query_phrase_idcs(
        keyword_list,
    )
    print(f" Total query phrases: {len(query_phrase_idcs)}")

    print(f"Saving keyword list with {len(keyword_list)} keywords to {KEYWORD_LIST}...")
    with open(KEYWORD_LIST, 'w', encoding='utf8') as f:
        json.dump(keyword_list, f, indent=4, ensure_ascii=False)

    print(f"Saving keyword test phrase indices to {KEYWORD_TESTPHRASE_RECORDS}...")
    with open(KEYWORD_TESTPHRASE_RECORDS, 'w', encoding='utf8') as f:
        f.write("\n".join(map(str, test_phrase_idcs)))

    print(f"Saving keyword query phrase indices to {KEYWORD_QUERY_RECORDS}...")
    with open(KEYWORD_QUERY_RECORDS, 'w', encoding='utf8') as f:
        f.write("\n".join(map(str, query_phrase_idcs)))

    print("\n✓ All keyword datafiles built successfully!")


def get_args() -> Namespace:
    parser = ArgumentParser(description='Build KWS lists and associated files')
    parser.add_argument(
        '--token_count', type=int, default=10,
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

