"""
Script to match each transcription in the dataset to the transcription
of the record with the highest speech-to-speech cosine similarity
(excluding itself).
"""

import pandas as pd
from constants import CSV_PATH, KEYPHRASE_PATH, SIMILARITY_MATRIX_PATH
from dataloading import load_tira_asr
import torch
from tqdm import tqdm
from pynini.lib.edit_transducer import LevenshteinDistance
from typing import *
from jiwer import wer, cer
import os
tqdm.pandas()

def get_most_similar_transcription_index(row, similarity_matrix):
    row_index = row.name
    similarities = similarity_matrix[row_index]
    similarities[row_index] = -1  # Exclude self-similarity
    most_similar_index = torch.argmax(similarities).item()
    return most_similar_index

def get_alphabet(df: pd.DataFrame) -> str:
    unique_chars = set()
    for transcription in df['transcription']:
        unique_chars.update(set(transcription))
    alphabet = ''.join(unique_chars)
    return alphabet

def get_edit_dist_funct(df: pd.DataFrame) -> Callable[[str, str], int]:
    alphabet = get_alphabet(df)
    levenshtein_dist = LevenshteinDistance(alphabet)
    dist_memo = {}
    def get_distance(s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        if (s1, s2) in dist_memo:
            return dist_memo[(s1, s2)]
        dist_memo[(s1, s2)] = levenshtein_dist.distance(s1, s2)
        return dist_memo[(s1, s2)]
    return get_distance

def main():
    dataset = load_tira_asr()
    dataset = dataset['train']
    df = dataset.to_pandas()

    similarity_matrix = torch.load(SIMILARITY_MATRIX_PATH)

    df['most_similar_index'] = df.progress_apply(
        lambda row: get_most_similar_transcription_index(row, similarity_matrix), axis=1
    )
    df['most_similar_transcription'] = df['most_similar_index'].progress_apply(
        lambda idx: df.iloc[idx]['transcription']
    )

    is_mismatched = df['transcription'] != df['most_similar_transcription']

    transcription_counts = df['transcription'].value_counts()
    transcription_list = transcription_counts.index.tolist()

    is_unique = df['transcription'].isin(
        transcription_counts[transcription_counts == 1].index
    )
    transcription_id = df['transcription'].apply(transcription_list.index)

    wer_values = df.progress_apply(
        lambda row: wer(
            row['transcription'],
            row['most_similar_transcription']
        ), axis=1
    )
    cer_values = df.progress_apply(
        lambda row: cer(
            row['transcription'],
            row['most_similar_transcription']
        ), axis=1
    )

    df['mismatched'] = is_mismatched
    df['unique'] = is_unique
    df['transcription_id'] = transcription_id
    df['wer'] = wer_values
    df['cer'] = cer_values


    num_nonunique = (~is_unique).sum()
    num_nonunique_mismatched = (is_mismatched & ~is_unique).sum()
    print(
        f"Number of non-unique transcriptions: {num_nonunique}"
    )
    print(
        f"Number of non-unique mismatched transcriptions: {num_nonunique_mismatched}"
    )
    print(f"{num_nonunique_mismatched / num_nonunique:.2%}",
          "of non-unique transcriptions are mismatched.")
    print(
        "Average WER for non-unique mismatched transcriptions:",
        f"{df.loc[is_mismatched & ~is_unique, 'wer'].mean():.2f}",
    )
    print(
        "Average CER for non-unique mismatched transcriptions:",
        f"{df.loc[is_mismatched & ~is_unique, 'cer'].mean():.2f}",
    )
    print(f"Overall average WER: {df['wer'].mean():.2f}")
    print(f"Overall average CER: {df['cer'].mean():.2f}")

    df['source'] = df['wav_source'].apply(
        os.path.basename
    ).apply(
        lambda f: os.path.splitext(f)[0]
    )

    rows_to_keep = [
        'transcription', 'most_similar_transcription',
        'mismatched', 'unique', 'wer', 'cer',
        'start', 'end', 'source', 'transcription_id',
    ]
    df = df[rows_to_keep]
    df['duration'] = df['end'] - df['start']

    df.to_csv(CSV_PATH, index_label='index')

    with open(KEYPHRASE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(transcription_list))

if __name__ == "__main__":
    main()