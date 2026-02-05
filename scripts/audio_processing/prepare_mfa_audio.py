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
from src.dataloading import load_tira_asr
import pandas as pd
import soundfile as sf

def main():
    args = get_args()
    
    # create MFA corpus directory structure
    speaker_dir = MFA_CORPUS_DIR / "himidan"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # load Tira ASR dataset
    print("Loading Tira ASR dataset...")
    tira_asr = load_tira_asr()

    # get indices of records to use based on KEYWORD_SENTENCES
    keyword_sentences_df = pd.read_csv(args.keyword_sentences_file)
    positive_mask = keyword_sentences_df['is_positive']
    record_indices = keyword_sentences_df[positive_mask]['record_idx'].tolist()

    # filter Tira ASR dataset to only include records with these indices
    tira_asr = tira_asr.select(record_indices)

    # save audio and transcription files in MFA format
    print(f"Saving audio and transcription files to {speaker_dir}...")
    def save_mfa_record(record, index):
        audio_path = speaker_dir / f"{index}.wav"
        transcription_path = speaker_dir / f"{index}.lab"

        # save audio file
        audio = record['audio']['array']
        sample_rate = record['audio']['sampling_rate']
        sf.write(audio_path, audio, sample_rate)

        # save transcription file (MFA expects a single line with the transcription
        with open(transcription_path, 'w') as f:
            f.write(record['transcription'])
    tira_asr.map(save_mfa_record, with_indices=True)
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