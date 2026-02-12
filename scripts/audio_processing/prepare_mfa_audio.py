"""
Prepare inputs for Montreal Forced Aligner (MFA) using Tira ASR dataset
audio and transcriptions. MFA expects the following corpus structure:
- corpus_dir/
    - speaker1/
        - utterance1.wav
        - utterance1.lab
        - utterance2.wav
        - utterance2.lab
Since the Tira ASR dataset has only one speaker (Himidan), we will
create a subdirectory `himidan/` within the MFA corpus directory and
place all audio and transcription files there.

Only uses records specified as positive in KEYWORD_SENTENCES, which
are the records used for keyword queries and test sentences in the
Interspeech 2026 experiment.
"""

from argparse import ArgumentParser
from src.constants import MFA_CORPUS_DIR, KEYWORD_SENTENCES
from src.dataloading import load_elicitation_cuts
import pandas as pd
import soundfile as sf

def main():
    args = get_args()
    
    # create MFA corpus directory structure
    speaker_dir = MFA_CORPUS_DIR / "himidan"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # load Tira supervisions
    print("Loading Tira supervisions...")
    cuts = load_elicitation_cuts()
    # trim to supervisions so we can filter by record index
    cuts = cuts.trim_to_supervisions()

    # get indices of records to use based on KEYWORD_SENTENCES
    keyword_sentences_df = pd.read_csv(args.keyword_sentences_file)
    positive_mask = keyword_sentences_df['is_positive']
    record_indices = keyword_sentences_df[positive_mask]['record_idx'].tolist()

    # filter Tira supervisions to only include records with these indices
    cuts = cuts.filter(lambda cut: int(getattr(cut, 'id')) in record_indices)
    cuts = cuts.to_eager()
    assert len(cuts) == len(record_indices), f"Expected {len(record_indices)} cuts "\
        + f"after filtering but got {len(cuts)}"

    # save audio and transcription files in MFA format
    print(f"Saving audio and transcription files to {speaker_dir}...")
    def save_mfa_record(cut):
        index = cut.id
        audio_path = speaker_dir / f"{index}.wav"
        transcription_path = speaker_dir / f"{index}.lab"

        # save audio file
        cut.save_audio(audio_path)

        # sanity check: ensure only one supervision
        assert len(cut.supervisions) == 1, f"Expected exactly one supervision for cut {cut.id}, but got {len(cut.supervisions)}"

        # save transcription file (MFA expects a single line with the transcription)
        with open(transcription_path, 'w') as f:
            f.write(cut.supervisions[0].custom['fst_text'])
    cuts.map(save_mfa_record)
    print("Finished preparing MFA corpus")

def get_args():
    parser = ArgumentParser(description="Prepare Tira ASR audio and transcriptions for MFA")
    parser.add_argument("--output_dir", type=str, default=MFA_CORPUS_DIR,
                        help="Directory to save prepared MFA corpus (default: %(default)s)")
    parser.add_argument("--keyword_sentences_file", type=str, default=KEYWORD_SENTENCES,
                        help="CSV file indicating Tira ASR records used for positive "\
                            +"and negative phrases (default: %(default)s)")

    return parser.parse_args()

if __name__ == "__main__":
    main()
