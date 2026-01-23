#!/usr/bin/env python3
"""
Build list for evaluating keyword search.
Requires `text_preproc` to be run first!

This script constructs the KEYWORD_LIST JSON file, which contains
information about keywords used for the Interspeech 2026 KWS experiment.
See `build_keyword_list` for JSON data structure.
"""

from collections import defaultdict
import pandas as pd
import random
import json
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Union
from unidecode import unidecode

# local imports
from src.constants import (
    WORDS_CSV, WORD2PHRASE_PATH, KEYWORD_LIST,
    RECORD2PHRASE_PATH, PHRASE_PATH,
    KEYWORD_POSITIVE_RECORDS, KEYWORD_NEGATIVE_RECORDS,
)

def build_keyword_list(
        word_df: pd.DataFrame,
        word2phrase: List[List[int]],
        phrase2records: List[List[int]],
        phrase_count: int=10,
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
        keywords: List of keyword strings
    """
    phrase_mask = word_df['phrase_count'] >= phrase_count
    print(f" Words with >= {phrase_count} phrases: {phrase_mask.sum()}")

    # apply unidecode so that combining diacritics don't affect length
    unidecode_words = word_df['word'].apply(unidecode)
    length_mask = unidecode_words.str.len() >= min_keyword_length
    print(f" Words with length >= {min_keyword_length}: {length_mask.sum()}")

    candidate_mask = phrase_mask & length_mask
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
        # sanity check to make sure we have enough phrases
        if len(phrase_idcs) < phrase_count:
            raise ValueError(
                f"Not enough phrases ({len(phrase_idcs)}) for word '{word}' "
                f"with phrase_count={phrase_count}"
            )

        # sample phrase_count phrases randomly
        random.Random(random_seed).shuffle(phrase_idcs)
        positive_phrase_idcs = phrase_idcs[:phrase_count]

        # get one positive record for each phrase
        positive_record_idcs = []
        for phrase_idx in positive_phrase_idcs:
            records = phrase2records[phrase_idx]
            if len(records) == 0:
                raise ValueError(
                    f"No records found for phrase index {phrase_idx} "
                    f"containing word '{word}'"
                )
            record_idx = random.Random(random_seed).choice(records)
            positive_record_idcs.append(record_idx)

        keyword_list.append({
            'keyword': word,
            'positive_phrase_idcs': positive_phrase_idcs,
            'positive_record_idcs': positive_record_idcs,
        })

    print(" Sampled keywords:")
    keywords = keyword_df['word'].tolist()
    print("\n".join(keywords))

    return keyword_list, keywords

def get_all_phrase_idcs(
        keywords: List[str],
        keyword_list: List[Dict[str, Union[str, List[int]]]],
        phrase_list: List[str],
        phrase2records: List[List[int]],
        negative_phrase_count: int=700,
        random_seed: int=1337,
    ) -> Tuple[List[int], List[int]]:
    """
    Get all positive and negative phrase indices for the given keywords.
    Positive indices are specified by `keyword_list`, negative indices are
    sampled randomly from phrases that do not contain any of the keywords.

    args:
        keywords: List of keyword strings
        keyword_list: List of keyword dicts
        phrase_list: List of all phrases
        phrase2records: Mapping from phrases to records containing the phrase
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
        record_idcs = phrase2records[idx]
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
    return list(positive_phrase_idcs), sampled_negative_idcs

def get_phrase2records(
        record2phrase: List[int],
) -> List[List[int]]:
    """
    Build mapping from phrase to records containing the phrase.
    Args:
        record2phrase: Mapping from records to phrase contained in the record
    Returns:
        phrase2records: Mapping from phrases to records containing the phrase
    """
    phrase2records: Dict[int, List[int]] = defaultdict(list)
    for record_idx, phrase_idx in enumerate(record2phrase):
        phrase2records[phrase_idx].append(record_idx)

    # convert to list of lists
    max_phrase_idx = max(phrase2records.keys())
    phrase2records_list: List[List[int]] = []
    for phrase_idx in range(max_phrase_idx + 1):
        records = phrase2records.get(phrase_idx, [])
        phrase2records_list.append(records)

    return phrase2records_list

def main():
    args = get_args()

    # Load files generated by `text_preproc.py`
    word_df = pd.read_csv(WORDS_CSV)
    print(f"Loaded {len(word_df)} unique words from {WORDS_CSV}")

    with open(WORD2PHRASE_PATH, 'r', encoding='utf8') as f:
        word2phrase = [list(map(int, line.strip().split())) for line in f.readlines()]
    print(f"Loaded word2phrase mapping from {WORD2PHRASE_PATH}")

    with open(RECORD2PHRASE_PATH, 'r', encoding='utf8') as f:
        record2phrase = [int(line.strip()) for line in f.readlines()]
    print(f"Loaded record2phrase mapping from {RECORD2PHRASE_PATH}")

    phrase2records = get_phrase2records(record2phrase)
    print("Built phrase2records mapping.")

    with open(PHRASE_PATH, 'r', encoding='utf8') as f:
        phrase_list = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(phrase_list)} phrases from {PHRASE_PATH}")

    print(f"\nBuilding keyword list with phrase_count={args.phrase_count} and num_keywords={args.num_keywords}...")

    # Build keyword list
    keyword_list, keywords = build_keyword_list(
        word_df,
        word2phrase,
        phrase2records,
        phrase_count=args.phrase_count,
        num_keywords=args.num_keywords,
        min_keyword_length=args.min_keyword_length,
        random_seed=args.random_seed,
    )

    # Get test phrase indices (positive + negative)
    positive_phrase_idcs, negative_phrase_idcs = get_all_phrase_idcs(
        keywords,
        keyword_list,
        phrase_list,
        phrase2records=phrase2records,
        negative_phrase_count=args.negative_phrase_count,
        random_seed=args.random_seed,
    )
    print(f" Total positive phrases: {len(positive_phrase_idcs)}")
    print(f" Total negative phrases: {len(negative_phrase_idcs)}")

    print(f"Saving keyword list with {len(keyword_list)} keywords to {KEYWORD_LIST}...")
    with open(KEYWORD_LIST, 'w', encoding='utf8') as f:
        json.dump(keyword_list, f, indent=4, ensure_ascii=False)

    print(f"Saving keyword negative phrase indices to {KEYWORD_NEGATIVE_RECORDS}...")
    with open(KEYWORD_NEGATIVE_RECORDS, 'w', encoding='utf8') as f:
        f.write("\n".join(map(str, negative_phrase_idcs)))

    print(f"Saving keyword positive phrase indices to {KEYWORD_POSITIVE_RECORDS}...")
    with open(KEYWORD_POSITIVE_RECORDS, 'w', encoding='utf8') as f:
        f.write("\n".join(map(str, positive_phrase_idcs)))

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

