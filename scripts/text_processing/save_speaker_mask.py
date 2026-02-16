"""
For a diarization dataset, save a speaker mask indicating which records correspond to speech from a target speaker.
For now, assuming that the dataset is TIRA DRZ.
"""

import argparse
from tira_kws.dataloading import load_tira_drz
from tira_kws.constants import LABELS_DIR
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Save speaker mask for TIRA DRZ dataset")
    parser.add_argument(
        '--speaker_id', '-s', type=str, required=True, help='Speaker ID to create mask for.'
    )
    return parser.parse_args()

def save_speaker_mask(speaker_id: str):
    print(f"Loading TIRA DRZ dataset...")
    ds = load_tira_drz()
    if 'train' in ds:
        ds = ds['train']

    speaker_col = 'tier'
    cols_to_drop = [col for col in ds.column_names if col != speaker_col]
    ds_speakers = ds.remove_columns(cols_to_drop)
    df = ds_speakers.to_pandas()
    speaker_mask = (df[speaker_col] == speaker_id).to_numpy()
    speaker_mask = torch.from_numpy(speaker_mask).to(torch.bool)
    mask_path = LABELS_DIR / f"speaker_mask_{speaker_id}.pt"
    print(f"Saving speaker mask for speaker {speaker_id} to {mask_path}...")
    torch.save(speaker_mask, mask_path)

def main():
    args = parse_args()
    save_speaker_mask(args.speaker_id)

if __name__ == "__main__":
    main()