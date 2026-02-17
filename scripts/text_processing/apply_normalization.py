#!/usr/bin/env python3
"""
# text_preproc.py
Depends on:
- Tira supervisions: JSONL manifest containing annotations for the Tira ASR dataset,
  see `src/dataloading.py` for loading code and `src/constants.py` for path constants.

Generates:
- MERGED_PHRASES_CSV: mapping of raw text to unicode-normalized text
- RECORD2PHRASE_PATH: mapping of record indices to phrase indices
- PHRASES_CSV: dataframe of unique phrases with token counts, gloss, and lemmata
- WORDS_CSV: dataframe of unique words with token counts, gloss and lemmata
- WORD2PHRASE_PATH: mapping of word indices to phrase indices containing them
"""

from tqdm import tqdm
from unidecode import unidecode
from tira_kws.constants import (
    WORDS_DIR, PHRASES_DIR,
    RECORDS_DIR, RECORD_LIST_CSV, UNNORMALIZED_WORDS
)
from tira_kws.dataloading import load_supervisions_df
import pandas as pd
from typing import Tuple

def build_merged_phrases_csv(df, output_path):
    """
    Build merged phrases CSV: mapping raw text to normalized text.
    Expects normalized text in column `text_normalized` and raw text in column `text`.
    Finds instances where normalization caused several dissimilar hand-transcribed
    sentences to merge.
    """
    print("Building merged phrases CSV...")

    # Build mapping from normalized text to raw text variants
    norm2raw = {}
    unique_normalized_texts = df['text_normalized'].unique().tolist()
    raw_strs_encountered = set()

    for text_normalized in tqdm(unique_normalized_texts, desc="Processing normalized texts"):
        mask = df['text_normalized'] == text_normalized
        text = df.loc[mask, 'text'].unique().tolist()
        norm2raw[text_normalized] = text
        # ensure only one normalized str per raw str
        assert not any(raw_str in raw_strs_encountered for raw_str in text)
        raw_strs_encountered.update(text)

    # Create dataframe with unique raw texts
    phrase_merges_df = df.drop_duplicates(subset=['text'])
    phrase_merges_df = phrase_merges_df.sort_values('text_normalized')
    phrase_merges_df = phrase_merges_df.reset_index(drop=True)

    phrase_merges_df['num_raw_variants'] = phrase_merges_df['text_normalized']\
        .apply(norm2raw.get)\
        .apply(len)
    phrase_merges_df = phrase_merges_df.sort_values('num_raw_variants', ascending=False)

    # Save to CSV
    phrase_merges_df.to_csv(output_path, index_label='index')
    print(f"Saved merged phrases CSV to {output_path}")
    print(f"  Total unique normalized texts:\t{len(unique_normalized_texts)}")
    print(f"  Total unique raw texts:\t{len(phrase_merges_df)}")

    return phrase_merges_df, norm2raw

def build_merged_words_csv(norm2raw, output_path):
    """
    Build merged words CSV: mapping raw word to normalized word.
    """
    print("Building merged words CSV...")

    # Create list of unique words and their normalized variants
    word_list = []
    for norm_text, raw_texts in norm2raw.items():
        for raw_text in raw_texts:
            # Split into words and add to list
            raw_words = raw_text.split()
            norm_words = norm_text.split()
            assert len(raw_words) == len(norm_words),\
                "Expected same number of raw and normalized words but got "\
                    + f"{len(raw_words)} raw and {len(norm_words)} normalized for text: "\
                    + f"{raw_text} -> {norm_text}"
            for raw_word, norm_word in zip(raw_words, norm_words):
                word_list.append({
                    'word': raw_word,
                    'word_normalized': norm_word
                })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(word_list)
    df = df.sort_values('word_normalized').reset_index(drop=True)
    df = df.drop_duplicates(subset=['word', 'word_normalized'])
    df['num_raw_variants'] = df.groupby('word_normalized')['word'].transform('nunique')
    df.to_csv(output_path, index=False)
    print(f"Saved merged words CSV to {output_path}")
    print(f"  Total unique normalized words:\t{len(df['word_normalized'].unique())}")
    print(f"  Total unique raw words:\t{len(df['word'].unique())}")

def join_aux_and_verbstem(normalized_text, gloss):
    """
    Example input:

        raw_text:           lálóvə́lɛ̂ðà                          nd̪ɔ̀bàgɛ̀
        normalized_text:    láló və́lɛ̀ðà                         nd̪ɔ̀bàgɛ̀
        gloss:              aux  pull[CLL,IT,IPFV,3PL.OBJ,aux]  tomorrow[]

    Output:
        "láló_və́lɛ̀ðà nd̪ɔ̀bàgɛ̀"
    """
    gloss_parts = gloss.split()
    normalized_text_parts = normalized_text.split()
    while "aux" in gloss_parts:
        aux_idx = gloss_parts.index("aux")
        # join aux with previous verb stem
        normalized_text_parts[aux_idx] = normalized_text_parts[aux_idx] + "_" + normalized_text_parts[aux_idx+1]
        del normalized_text_parts[aux_idx+1]
        del gloss_parts[aux_idx]

    return " ".join(normalized_text_parts)

def filter_stranded_diacritics(raw_text: str, normalized_text: str) -> Tuple[str, str]:
    raw_words = raw_text.split()
    norm_words = normalized_text.split()

    raw_words_filtered = []
    norm_words_filtered = []

    for raw_word, norm_word in zip(raw_words, norm_words):
        if not unidecode(raw_word).strip():
            continue
        raw_words_filtered.append(raw_word)
        norm_words_filtered.append(norm_word)
    raw_text_filtered = " ".join(raw_words_filtered)
    norm_text_filtered = " ".join(norm_words_filtered)
    return raw_text_filtered, norm_text_filtered

def main():
    # Ensure output directory exists
    WORDS_DIR.mkdir(parents=True, exist_ok=True)
    PHRASES_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Tira ASR dataset...")
    df = load_supervisions_df()
    print(f"Loaded dataset with {len(df)} records")
    print(df.head())

    # perform preprocessing on FST normalized text before building merges
    df = df.rename(columns={'fst_text': 'text_normalized'})

    # first join auxiliary verbs with verb stems in the normalized text
    # so that we don't end up with cases where the raw text has one word
    # but the normalized text has two
    print("Joining auxiliary verbs with verb stems in normalized text...")
    df['text_normalized'] = df.apply(
        lambda row: join_aux_and_verbstem(row['text_normalized'], row['gloss']),
        axis=1
    )

    # also filter out any word where the unnormalized form is just a
    # diacritic with no actual characters
    print("Filtering out stranded diacritics...")
    filtered_texts = df.apply(
        lambda row: filter_stranded_diacritics(row['text'], row['text_normalized']),
        axis=1
    )
    df['text'] = filtered_texts.apply(lambda x: x[0])
    df['text_normalized'] = filtered_texts.apply(lambda x: x[1])

    print("Building merges using FST normalization...")
    fst_norm_phrases_path = PHRASES_DIR / "merges_fst.csv"
    fst_norm_words_path = WORDS_DIR / "merges_fst.csv"

    _, norm2raw = build_merged_phrases_csv(df, fst_norm_phrases_path)
    build_merged_words_csv(norm2raw, fst_norm_words_path)

    print("\nBuilding merges using unidecode normalization...")
    df['fst_normalized'] = df['text_normalized']
    df['text_normalized'] = df['text'].apply(lambda x: unidecode(x))
    df['text_normalized'] = df['text_normalized'].str.strip()
    df['text_normalized'] = df['text_normalized'].str.lower()

    unidecode_norm_phrases_path = PHRASES_DIR / "merges_unidecode.csv"
    unidecode_norm_words_path = WORDS_DIR / "merges_unidecode.csv"

    _, norm2raw = build_merged_phrases_csv(df, unidecode_norm_phrases_path)
    build_merged_words_csv(norm2raw, unidecode_norm_words_path)

    df['unidecode_normalized'] = df['text_normalized']
    df = df.drop(columns=['text_normalized'])

    # save updated dataframe to use for building other files
    df.to_csv(RECORD_LIST_CSV, index_label='record_idx')
    unnormalized_words = set()
    df['text'].str.split().apply(unnormalized_words.update)
    with open(UNNORMALIZED_WORDS, 'w') as f:
        for word in sorted(unnormalized_words):
            f.write(word + "\n")
    print(f"Saved updated supervisions dataframe to {RECORD_LIST_CSV}")
    print(f"Saved list of unique unnormalized words to {UNNORMALIZED_WORDS}")

if __name__ == "__main__":
    main()