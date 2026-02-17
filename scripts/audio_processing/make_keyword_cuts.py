"""
Build Lhotse cutset of keywords for KWS evaluation.
Segment keywords from sentence audio using timestamps from MFA output.

This script adds the `keyword_idcs` key to the existing KEYWORD_LIST JSON file.
See JSON structure in `keyword_list_builder.py`.
"""

from tira_kws.constants import (
    MFA_SPEAKER_OUTPUT_DIR, KEYWORD_MANFIEST, WORD_MERGES_CSV,
    KEYWORD_SENTENCES, KEYWORDS_CSV, RECORDING_MANIFEST,
    RECORD_LIST_CSV, NORMALIZATION_TYPE,
)
from tira_kws.dataloading import load_elicitation_cuts
from tgt.io import read_textgrid
import pandas as pd
import argparse
from pathlib import Path
from lhotse import SupervisionSegment, SupervisionSet, RecordingSet
from lhotse.supervision import AlignmentItem
from lhotse.qa import validate_recordings_and_supervisions
from unicodedata import normalize
from typing import List, Tuple

def main():
    args = get_args()

    # load Tira supervisions
    print("Loading Tira supervisions...")

    keyword_sentences_df = pd.read_csv(args.keyword_sentences_file)
    keyword_df = pd.read_csv(KEYWORDS_CSV, keep_default_na=False)
    merges_df = pd.read_csv(WORD_MERGES_CSV, keep_default_na=False)
    record_df= pd.read_csv(RECORD_LIST_CSV, keep_default_na=False)

    record_indices = keyword_sentences_df['record_idx'].tolist()
    normalized_col = f"{NORMALIZATION_TYPE}_normalized"
    cuts = load_elicitation_cuts(index_list=record_indices)

    positive_indices = keyword_sentences_df[keyword_sentences_df['is_positive']]['record_idx'].apply(str).tolist()

    # save keyword and test phrase supervisions to a new set
    kws_supervisions = []

    def add_supervisions(cut: SupervisionSegment):
        # sanity check: should only be one supervision per cut
        assert len(cut.supervisions) == 1, f"Expected exactly one supervision for cut {cut.id}, but got {len(cut.supervisions)}"
        sentence_supervision = cut.supervisions[0]
        sentence_supervision.start = cut.start

        # set text to string from record list
        # so that preprocessing is accounted for
        record_mask = record_df['record_idx'] == int(cut.id)
        if record_mask.sum() != 1:
            raise ValueError(f"Expected exactly one record corresponding to cut {cut.id} in record list, but got {record_df[record_mask]}")
        sentence_supervision.text = str(record_df.loc[record_mask, 'text'].iloc[0])
        sentence_supervision.custom['normalized_text'] = str(record_df.loc[record_mask, normalized_col].iloc[0])

        # only add keyword supervisions for cuts corresponding to positive records
        # since negative records don't contain the keyword
        if cut.id in positive_indices:
            word_idx, current_keyword = get_current_keyword(keyword_sentences_df, keyword_df, cut)
            raw_keywords = get_raw_keywords(current_keyword, merges_df)
            keyword_alignment = get_keyword_alignment(raw_keywords, cut)

            sentence_supervision.custom['keyword_type'] = "positive_sentence"
            sentence_supervision.custom['keyword'] = current_keyword
            sentence_supervision.custom['keyword_id'] = word_idx
            sentence_supervision.alignment = {'word': [keyword_alignment]}
        else:
            sentence_supervision.custom['keyword_type'] = "negative_sentence"

        kws_supervisions.append(sentence_supervision)
    
    cuts.map(add_supervisions) # type: ignore

    supervision_set = SupervisionSet.from_segments(kws_supervisions)


    print("Validating supervision manifest against recordings...")
    recording_set = RecordingSet.from_jsonl(RECORDING_MANIFEST)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    # save new supervision set to manifest
    print(f"Saving keyword cut manifest to {args.output}...")
    supervision_set.to_file(args.output)

def get_current_keyword(keyword_sentences_df, keyword_df, cut) -> Tuple[int, str]:
    word_idx = keyword_sentences_df.loc[
            keyword_sentences_df['record_idx'] == int(cut.id), 'word_idx'
        ]
    if len(word_idx) != 1:
        raise ValueError(f"Expected exactly one word index for record {cut.id}, but got {word_idx}")
    word_idx = int(word_idx.iloc[0])

    current_keyword = keyword_df.loc[
            keyword_df['word_idx'] == word_idx, 'word'
        ]
    if len(current_keyword) != 1:
        raise ValueError(f"Expected exactly one keyword for record {cut.id}, but got {current_keyword}")
    current_keyword = str(current_keyword.iloc[0])
    return word_idx, current_keyword

def get_raw_keywords(current_keyword: str, merges_df: pd.DataFrame) -> List[str]:
    """
    Get all 'word' entries corresponding to the given normalized keyword
    ('word_normalized' column) from the merges dataframe.
    """
    normalized_word_mask = merges_df['word_normalized'] == current_keyword
    if normalized_word_mask.sum() == 0:
        raise ValueError(f"Expected at least one raw word corresponding to normalized keyword '{current_keyword}', but got {merges_df[normalized_word_mask]}")
    raw_keywords = merges_df.loc[normalized_word_mask, 'word'].tolist()
    return raw_keywords

def get_keyword_alignment(
          current_keywords: List[str],
          cut: SupervisionSegment
        ) -> AlignmentItem:
    record_id = cut.id


    textgrid_path = MFA_SPEAKER_OUTPUT_DIR / f"{record_id}.TextGrid"
    textgrid = read_textgrid(str(textgrid_path))
    word_tier = textgrid.get_tier_by_name("words")
    intervals = word_tier.annotations
    keyword_interval = None
    for interval in intervals:
        # normalize text to ensure matching with keywords in KEYWORDS_CSV
        interval_text = normalize("NFKD", interval.text)
        if interval_text in current_keywords:
            keyword_interval = interval
            break

    # sanity check: ensure at least one interval with the keyword text
    assert keyword_interval is not None, "Expected at least one interval with keyword"\
        + f" '{current_keywords}' for record {record_id}, but got {intervals}"

    # start and end times of the keyword interval are relative to the start of the cut
    # change to be absolute times relative to the start of the recording
    start = cut.start + keyword_interval.start_time
    end = cut.start + keyword_interval.end_time
    # create supervision segment for the keyword interval
    keyword_alignment = AlignmentItem(
        symbol=keyword_interval.text,
        start=start,
        duration=end-start,
    )
    return keyword_alignment

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add word alignments from MFA to lhotse supervision set")
    parser.add_argument("--output", type=Path, default=KEYWORD_MANFIEST,
                        help="Filepath to save keyword cut manifest (default: %(default)s)")
    parser.add_argument("--keyword_sentences_file", type=Path, default=KEYWORD_SENTENCES,
                        help="CSV file indicating Tira ASR records used for positive "\
                            +"and negative phrases (default: %(default)s)")

    return parser.parse_args()

if __name__ == '__main__':
    main()