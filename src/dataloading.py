from src.constants import (
    SUPERVISION_MANIFEST, RECORDING_MANIFEST
)

from typing import *
import pandas as pd
from lhotse import CutSet, RecordingSet, SupervisionSet

def load_supervisions_df() -> pd.DataFrame:
    """
    Load supervision segments from the Tira elicitation dataset
    and return as a dataframe.
    """
    supervisions_df = pd.read_json(SUPERVISION_MANIFEST, lines=True)
    supervisions_df = supervisions_df.set_index('id')

    # the columns 'fst_text', 'gloss' and 'root' are nested under 'custom'
    supervisions_df['fst_text'] = supervisions_df['custom'].apply(lambda x: x['fst_text'])
    supervisions_df['gloss'] = supervisions_df['custom'].apply(lambda x: x['gloss'])
    supervisions_df['lemmata'] = supervisions_df['custom'].apply(lambda x: x['root'])
    supervisions_df = supervisions_df.drop(columns=['custom'])

    return supervisions_df

def load_elicitation_cuts(index_list: Optional[List[int]] = None) -> CutSet:
    recordings = RecordingSet.from_jsonl(RECORDING_MANIFEST)
    supervisions = SupervisionSet.from_jsonl(SUPERVISION_MANIFEST)
    cuts = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions,
    )

    if index_list is not None:
        # since index list corresponds to record indices, first flatten
        # cuts to supervisions so we can filter by record index
        cuts = cuts.trim_to_supervisions()
        cuts = cuts.filter(lambda cut: int(getattr(cut, 'id')) in index_list)
        cuts = cuts.to_eager()
        assert len(cuts) == len(index_list), f"Expected {len(index_list)} cuts after filtering but got {len(cuts)}"

    return cuts