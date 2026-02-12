#!/usr/bin/env python3
"""
# text_preproc.py
Depends on:
- Tira supervisions: JSONL manifest containing annotations for the Tira ASR dataset,
  see `src/dataloading.py` for loading code and `src/constants.py` for path constants.

Generates:
- MERGED_PHRASES_CSV: mapping of EAF text to FST-normalized text
- RECORD2PHRASE_PATH: mapping of record indices to phrase indices
- PHRASES_CSV: dataframe of unique phrases with token counts, gloss, and lemmata
- WORDS_CSV: dataframe of unique words with token counts, gloss and lemmata
- WORD2PHRASE_PATH: mapping of word indices to phrase indices containing them
"""

from tqdm import tqdm
import numpy as np
from src.constants import (
    WORDS_DIR, PHRASES_DIR, MERGED_PHRASES_CSV, PHRASES_CSV,
    PHRASE2RECORDS_PATH, WORD2PHRASE_PATH, WORDS_CSV
)
from src.dataloading import load_supervisions_df
import pandas as pd

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
    unique_phrase_df = unique_phrase_df.rename(columns={'fst_text':'phrase'})

    # Exclude phrases with only one word
    single_word_mask = unique_phrase_df['phrase'].str.split().apply(len) <= 1
    num_single_words = single_word_mask.sum()
    unique_phrase_df = unique_phrase_df[~single_word_mask]
    print(f"Found {num_single_words} phrases with only one word, dropping")


    # Count occurrences of each phrase
    token_counts = df['fst_text'].value_counts()

    # Map token counts to phrases
    unique_phrase_df = unique_phrase_df.set_index('phrase')
    unique_phrase_df['token_count'] = token_counts
    unique_phrase_df = unique_phrase_df.reset_index()

    print(f"  Total unique phrases: {len(unique_phrase_df)}")
    print(f"  Token count statistics:\n{unique_phrase_df['token_count'].describe()}")

    unique_phrase_df.to_csv(output_path, index_label='index')
    print(f"Saved phrases CSV to {output_path}")

    return unique_phrase_df

def build_words_csv(unique_phrase_df, output_path):
    """
    Build WORDS_CSV: dataframe with unique FST-normalize words alongside
    token and phrase counts.
    """
    print("\nBuilding words CSV...")

    # Map words to lemmata
    word2lemma = {}
    def update_word2lemma(row):
        words = row['phrase'].split()
        lemmata = row['lemmata'].split()
        for word, lemma in zip(words, lemmata):
            if word in word2lemma:
                # concatenate lemmata
                lemma_parts = word2lemma[word].split(', ')
                lemma_parts = list(map(str.strip, lemma_parts))
                if lemma in lemma_parts:
                    continue
                word2lemma[word] = ', '.join(lemma_parts+[lemma])
            else:
                word2lemma[word] = lemma
    unique_phrase_df.apply(update_word2lemma, axis=1)

    unique_words = list(word2lemma.keys())
    unique_lemmata = set(word2lemma.values())

    avg_words_per_lemma = len(unique_words) / len(unique_lemmata)

    # Get token and phrase counts per word
    word_rows = []
    for word in unique_words:
        word_mask = unique_phrase_df['phrase'].str.contains(word)
        token_count = unique_phrase_df.loc[word_mask, 'token_count'].sum()
        word_rows.append({
            'word': word,
            'token_count': token_count,
            'lemma': word2lemma[word],
            'phrase_count': word_mask.sum(),
        })

    word_df = pd.DataFrame(data=word_rows)

    print(f"  Total unique words: {len(word_df)}")
    print(f"  Total unique lemmata: {len(unique_lemmata)}")
    print(f"  Avg words per lemma: {avg_words_per_lemma}")
    print(f"  Token count statistics:\n{word_df['token_count'].describe()}")
    print(f"  Phrase count statistics:\n{word_df['phrase_count'].describe()}")
    word_df.to_csv(output_path, index_label='index')
    print(f"Saved words CSV to {output_path}")

    return word_df

def build_phrase2records(df, unique_phrase_df, output_path) -> pd.DataFrame:
    """
    Build RECORD2PHRASE_PATH: mapping of record index to phrase index.
    """
    print("\nBuilding record2phrase mapping...")

    # Create mapping from phrase to index
    phrase_to_idx = {phrase: idx for idx, phrase in unique_phrase_df['phrase'].items()}

    # Map each record to its phrase index
    record2phrase = df['fst_text'].apply(phrase_to_idx.get).tolist()

    # Save to dataframe
    record2phrase_df = pd.DataFrame(record2phrase, columns=['phrase_idx'])
    record2phrase_df.to_csv(output_path, index_label='record_idx')

    print(f"Saved record2phrase mapping to {output_path}")
    print(f"  Total records: {len(record2phrase)}")

    return record2phrase

def build_word2phrase(words_df, unique_phrase_df, output_path) -> pd.DataFrame:
    """
    Build WORD2PHRASE_PATH: mapping of word index to indices
    of phrases containing the word.
    """
    print("\nBuilding word2phrase mapping...")

    # Create mapping from word to phrase indices
    rows = []
    for index, word in words_df['word'].items():
        phrase_mask = unique_phrase_df['phrase'].str.split()\
            .apply(lambda words: word in words)
        phrase_indices = unique_phrase_df.index[phrase_mask].tolist()
        rows.extend({
            'word_idx': index,
            'phrase_idx': phrase_idx
        } for phrase_idx in phrase_indices)
    word2phrase_df = pd.DataFrame(rows)

    # Save to file
    word2phrase_df.to_csv(output_path, index=False)

    print(f"Saved word2phrase mapping to {output_path}")
    print(f"  Total records: {len(word2phrase_df)}")

    return word2phrase_df

def main():
    # Ensure output directory exists
    WORDS_DIR.mkdir(parents=True, exist_ok=True)
    PHRASES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Tira ASR dataset...")
    df = load_supervisions_df()
    df = df.rename(columns={'text': 'eaf_text'})
    df = df.rename_axis('record_id', axis=1)
    print(f"Loaded dataset with {len(df)} records")
    print(df.head())

    # Build all files
    eaf_unique_df = build_merged_phrases_csv(df, MERGED_PHRASES_CSV)
    unique_phrase_df = build_phrases_csv(eaf_unique_df, df, PHRASES_CSV)
    words_df = build_words_csv(unique_phrase_df, WORDS_CSV)

    build_phrase2records(df, unique_phrase_df, PHRASE2RECORDS_PATH)
    build_word2phrase(words_df, unique_phrase_df, WORD2PHRASE_PATH)

if __name__ == '__main__':
    main()
