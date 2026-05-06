"""
Save a Lhotse cut manifest with supervisions for KWS sentences from CSS Capstone data.
The output manifest has the same data as `supservisions.jsonl` with the addition
of the 'keyword' and 'record_type' fields, where 'keyword' is a string indicating
the keyword and 'record_type' is one of 'positive', 'close_negative', 'negative'.
"""

from tira_kws.constants import (
    RECORDING_MANIFEST,
    CAPSTONE_SUPERVISIONS,
    CAPSTONE_POSITIVE_RECORDS,
    CAPSTONE_CLOSE_NEGATIVE_RECORDS,
    CAPSTONE_NEGATIVE_RECORDS,
)
from tira_kws.dataloading import load_elicitation_cuts
from functools import cache
import pandas as pd
from typing import Literal
from lhotse.cut import CutSet, Cut
from lhotse import SupervisionSegment, SupervisionSet, RecordingSet
from lhotse.qa import validate_recordings_and_supervisions
from copy import deepcopy
from tqdm import tqdm

tqdm.pandas()


def main():
    print("Loading elicitation cuts...")
    source_cuts = load_elicitation_cuts()
    source_cuts = source_cuts.trim_to_supervisions()

    print("Reading keyword lists...")
    positive_df = pd.read_csv(CAPSTONE_POSITIVE_RECORDS)
    negative_df = pd.read_csv(CAPSTONE_NEGATIVE_RECORDS)
    close_negative_df = pd.read_csv(CAPSTONE_CLOSE_NEGATIVE_RECORDS)

    @cache
    def get_source_cut(id: int) -> Cut:
        return get_cut(source_cuts, id)

    def make_keyword_supervision(
        row: dict,
        record_type: Literal["positive", "close_negative", "negative"],
    ) -> Cut:
        id = row["sentence_id"]
        cut = get_source_cut(id)
        cut = deepcopy(cut)
        supervision = cut.supervisions[0]
        if record_type == "negative":
            supervision_id = id
        else:
            supervision_id = f"{row['keyword']}_{id}"
            supervision.custom["keyword"] = row["keyword"]
        supervision.custom["norm_text"] = row["final_sentence"]
        supervision_id = f"{supervision_id}_{record_type}"
        supervision.id = supervision_id
        supervision.custom["record_type"] = record_type

        cut.id = supervision_id
        cut.custom = supervision.custom

        return cut

    print("Saving positive supervisions...")
    positive_cuts = positive_df.progress_apply(
        lambda row: make_keyword_supervision(row, record_type="positive"),
        axis=1,
    ).tolist()

    print("Saving close negative supervisions...")
    close_negative_cuts = close_negative_df.progress_apply(
        lambda row: make_keyword_supervision(row, record_type="close_negative"),
        axis=1,
    ).tolist()

    print("Saving negative supervisions...")
    negative_cuts = negative_df.progress_apply(
        lambda row: make_keyword_supervision(row, record_type="negative"),
        axis=1,
    ).tolist()

    kws_cutset = CutSet.from_cuts(positive_cuts + close_negative_cuts + negative_cuts)

    print("Validating supervision manifest against recordings...")
    recording_set = RecordingSet.from_jsonl(RECORDING_MANIFEST)
    validate_recordings_and_supervisions(recording_set, kws_cutset)

    # save new supervision set to manifest
    print(f"Saving keyword cut manifest to {CAPSTONE_SUPERVISIONS}")
    kws_cutset.to_file(CAPSTONE_SUPERVISIONS)


def get_cut(cuts: CutSet, id: int) -> SupervisionSegment:
    filtered_cut = cuts.filter(lambda cut: int(cut.id) == id)
    filtered_cut = filtered_cut.to_eager()
    if not filtered_cut:
        raise KeyError(f"Record {id} not found in cutset")

    return filtered_cut[0]


if __name__ == "__main__":
    main()
