from importlib.util import find_spec
import torch
import os
from pathlib import Path

# audio and model constants

## hyperparameters

SAMPLE_RATE = 16_000
DEVICE = torch.device(0 if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))

## model names

CLAP_IPA_ENCODER_NAME = 'anyspeech/clap-ipa-{encoder_size}-speech'
CLAP_IPA_TEXT_ENCODER_NAME = 'anyspeech/clap-ipa-{encoder_size}-phone'
IPA_ALIGNER_ENCODER_NAME = 'anyspeech/ipa-align-{encoder_size}-speech'

SPEECHBRAIN_LID_ENCODER_NAME = 'speechbrain/lang-id-voxlingua107-ecapa'

# file paths

## dataset paths
DATA_DIR = Path(os.environ.get("DATASETS", os.path.expanduser("~/datasets")))
TIRA_ASR_PATH = DATA_DIR / "tira_asr"
TIRA_DRZ_PATH = DATA_DIR / "tira_drz"

## local data paths
LABELS_DIR = Path("data/labels")
PHRASELIST_PATH = LABELS_DIR / "tira_keyphrases.txt"
KEYPHRASE_CSV = LABELS_DIR / "tira_keyphrases.csv"

PREDICTED_WORDS_CSV = Path("data/kws_predictions/most_predicted_word.csv")
SIMILARITY_MATRIX_PATH = Path("data/similarity_matrix/similarity_matrix.pt")
EMBEDDINGS_DIR = DATA_DIR/"tira_kws"/"embeddings"

# keyword constants
MAX_KEYWORD_STR = '$max'
MEAN_KEYWORD_STR = '$mean'
CLAP_IS_AVAILABLE = find_spec('clap') is not None
SPEECHBRAIN_IS_AVAILABLE = find_spec('speechbrain') is not None