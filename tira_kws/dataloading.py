from tira_kws.constants import (
    SUPERVISION_MANIFEST, RECORDING_MANIFEST, FEATURES_DIR
)

from typing import *
import pandas as pd
from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.cut import Cut
from lhotse.dataset import K2SpeechRecognitionDataset, DynamicBucketingSampler
from torch.utils.data import DataLoader
from argparse import Namespace


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

    # rename index to record_id for clarity and consistency with other files
    supervisions_df = supervisions_df.rename_axis('record_id', axis=1)

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

def load_kws_cuts(
        feature_set: Literal['xlsr', 'fbank'],
        normalize_text: bool = False,
) -> Tuple[CutSet, CutSet]:
    cuts_path = FEATURES_DIR / f"kws_{feature_set}.jsonl"
    cuts = CutSet.from_jsonl(cuts_path)
    
    word_cuts = cuts.trim_to_alignments(type='word')
    sentence_cuts = cuts.trim_to_supervisions()

    if normalize_text:
        # 'normalized_text' field should be populated in supervisions already
        def normalize_cut_text(cut: Cut) -> Cut:
            cut.supervisions[0].text = cut.supervisions[0].custom['normalized_text']
            return cut
        word_cuts = word_cuts.map(normalize_cut_text)
        sentence_cuts = sentence_cuts.map(normalize_cut_text)
    return word_cuts, sentence_cuts

def get_k2_dataloader(cuts: CutSet, args: Namespace) -> DataLoader:
    # based on GigaSpeechAsrDataModule.test_dataloders()
    # in icefall/egs/gigaspeech/KWS/zipformer/asr_datamodule.py
    # hardcoding defaults based on values in icefall/egs/gigaspeech/KWS/run.sh
    test_ds = K2SpeechRecognitionDataset(return_cuts=True)
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=args.max_duration,
        max_cuts=args.max_cuts,
        shuffle=False
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=None,
        sampler=sampler,
        num_workers=0,
    )
    return test_dl