#!/usr/bin/env python3
"""
- MERGED_PHRASES_CSV: mapping of EAF text to FST-normalized text
- PHRASE_PATH: text file with all unique phrases
- RECORD2PHRASE_PATH: mapping of record indices to phrase indices
- PHRASES_CSV: dataframe of unique phrases with token counts
"""

from tqdm import tqdm
import numpy as np
from src.constants import (
    LABELS_DIR, MERGED_PHRASES_CSV, PHRASES_CSV,
    PHRASE_PATH, RECORD2PHRASE_PATH, WORD2PHRASE_PATH, WORD_PATH, WORDS_CSV
)
from src.dataloading import load_tira_asr
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


def build_phrase_list(unique_phrase_df, output_path):
    """
    Build PHRASE_PATH: text file with all unique phrases (one per line).
    """
    print("\nBuilding phrase list...")

    all_phrases = unique_phrase_df['phrase'].tolist()

    with open(output_path, 'w', encoding='utf8') as f:
        for phrase in all_phrases:
            f.write(phrase + '\n')

    print(f"Saved phrase list to {output_path}")
    print(f"  Total phrases: {len(all_phrases)}")

    return all_phrases

def build_words_csv(unique_phrase_df, output_path):
    """
    Build WORDS_CSV: dataframe with unique FST-normalize words and token counts.
    """
    print("\nBuilding words CSV...")

    # Get all unique words
    unique_words = set()
    unique_phrase_df['phrase'].str.split().apply(unique_words.update)


    # Get token counts per word
    word_rows = []
    for word in unique_words:
        word_mask = unique_phrase_df['phrase'].str.contains(word)
        token_count = unique_phrase_df.loc[word_mask, 'token_count'].sum()
        word_rows.append({
            'word': word,
            'token_count': token_count
        })

    word_df = pd.DataFrame(data=word_rows)

    print(f"  Total unique words: {len(word_df)}")
    print(f"  Token count statistics:\n{word_df['token_count'].describe()}")

    word_df.to_csv(output_path, index_label='index')
    print(f"Saved words CSV to {output_path}")

    return word_df

def build_word_list(word_df, output_path):
    """
    Build WORD_PATH: text file with all unique words (one per line).
    """
    print("\nBuilding word list...")

    all_words = word_df['word'].tolist()

    with open(output_path, 'w', encoding='utf8') as f:
        for word in all_words:
            f.write(word + '\n')

    print(f"Saved wordlist to {output_path}")
    print(f"  Total words: {len(all_words)}")

    return all_words

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

def build_word2phrase(all_words, unique_phrase_df, output_path):
    """
    Build WORD2PHRASE_PATH: mapping of word index to indices
    of phrases containing the word.
    """
    print("\nBuilding word2phrase mapping...")

    # Create mapping from word to phrase indices
    word2phrases = []
    for word in all_words:
        phrase_mask = unique_phrase_df['phrase'].str.contains(word)
        phrase_indices = unique_phrase_df.index[phrase_mask].tolist()
        word2phrases.append(phrase_indices)

    # Save to file
    with open(output_path, 'w', encoding='utf8') as f:
        for phrase_indices in word2phrases:
            phrase_indices_str = ' '.join(map(str, phrase_indices))
            f.write(phrase_indices_str+"\n")

    print(f"Saved word2phrase mapping to {output_path}")
    print(f"  Total records: {len(word2phrases)}")

    return word2phrases

def main():
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
    df: pd.DataFrame = ds_noaudio.to_pandas() # type: ignore
    df = df.rename(columns=colmap)
    print(f"DataFrame shape: {df.shape}")

    # Build all files
    eaf_unique_df = build_merged_phrases_csv(df, MERGED_PHRASES_CSV)

    unique_phrase_df = build_phrases_csv(eaf_unique_df, df, PHRASES_CSV)

    all_phrases = build_phrase_list(unique_phrase_df, PHRASE_PATH)
    record2phrase = build_record2phrase(df, all_phrases, RECORD2PHRASE_PATH)

    words_df = build_words_csv(unique_phrase_df, WORDS_CSV)
    all_words = build_word_list(words_df, WORD_PATH)
    word2phrases = build_word2phrase(all_words, unique_phrase_df, WORD2PHRASE_PATH)

if __name__ == '__main__':
    main()
