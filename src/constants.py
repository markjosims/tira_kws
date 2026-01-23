from importlib.util import find_spec
import torch
import os
from pathlib import Path

from wandb.env import DATA_DIR

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
TIRA_ASR_URI = "css-kws-capstone/tira-asr"

## local data paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
LABELS_DIR = DATA_DIR / "labels"
CONFIG_DIR = PROJECT_DIR / "config"

"""
### source data for KWS
- MERGED_PHRASES_CSV: dataframe mapping `eaf_text` (raw string from
    ELAN) to `fst_text` (string with most likely normalized form
    using FST parser).
- PHRASES_CSV: dataframe where each row is a unique phrase (only
    uses FST-normalized text) along with its token count.
- KEYPHRASE_CSV: similar to `PHRASES_CSV` but also including columns
    for which phrases are used as keyphrase queries and how many
    easy/medium/hard negative records exist for a given phrase.
- WORDS_CSV: similar to `PHRASES_CSV` except that each row is a unique
    word rather than phrase
- CER_MATRIX_PATH: matrix of CER values of keyphrases (rows) to all
    phrases (columns).
- PHRASE_PATH: all unique Tira phrases
- WORD_PATH: all unique Tira words
- RECORD2PHRASE_PATH: phrase index for each record
- WORD2PHRASE_PATH: indices of phrases containing each word
"""
MERGED_PHRASES_CSV = LABELS_DIR / "keyphrases_rewritten_merges.csv"
PHRASES_CSV = LABELS_DIR / "tira_phrases.csv"
KEYPHRASE_CSV = LABELS_DIR / "keyphrases.csv"
WORDS_CSV = LABELS_DIR / "tira_words.csv"
CER_MATRIX_PATH = LABELS_DIR / "cer_matrix.np"
PHRASE_PATH = LABELS_DIR / "tira_phrases.txt"
WORD_PATH = LABELS_DIR / "tira_words.txt"
RECORD2PHRASE_PATH = LABELS_DIR / "record2phrase.txt"
WORD2PHRASE_PATH = LABELS_DIR / "word2phrase.txt"


"""
### Keyword lists
Lists related to keyword queries for the Interspeech 2026 experiment.

- KEYWORD_LIST: JSON list for all keywords. See `keyword_list_builder.py`
    and `make_keyword_ds.py` for details.
- KEYWORD_TESTPHRASE_RECORDS: indices of all Tira records used as test phrases
- KEYWORD_QUERY_RECORDS: indices of all Tira records used as keyword queries

"""
KEYWORD_LIST = LABELS_DIR / "keyword_list.json"
KEYWORD_TESTPHRASE_RECORDS = LABELS_DIR / "keyword_testphrase_idcs.txt"
KEYWORD_QUERY_RECORDS = LABELS_DIR / "keyword_query_idcs.txt"

"""
### Keyphrase lists
Lists related to keyphrases used for the Sparse Annotation Filling experiment.

- KEYPHRASE_PATH: indices of all Tira phrases used as keyphrase
    queries
- KEYPHRASE_LIST: JSON list for all keyphrases including positive
    and negative records. See below for JSON data structure.
- CALIBRATION_LIST: JSON list with same structure as
    `KEYPHRASE_LIST`. This list maps a balanced subset of
    keyphrases to negative records, and is used for tuning the
    detection threshold for KWS.
- ENGLISH_CALIBRATION_LIST: Indices of English keyphrases used for
    KWS calibration.

```json
[
    {
        'keyphrase': $str,
        'keyphrase_idx': $int,
        'record_idcs': [$int, $int, ...]
        'easy': [$int, $int, ...]
        'medium': [$int, $int, ...]
        'hard': [$int, $int, ...]
    },
    ...
]
```

Where the 'easy', 'medium' and 'hard' keys map to lists of
record indices. This list maps all keyphrases to all positive
and negative records.
"""

KEYPHRASE_PATH = LABELS_DIR / "tira_keyphrase_idcs.txt"
KEYPHRASE_LIST = LABELS_DIR / "keyphrase_list.json"
CALIBRATION_LIST = LABELS_DIR / "calibration_list.json"
ENGLISH_CALIBRATION_LIST = LABELS_DIR / "english_calibration_keyphrase_idcs.txt"

"""
### output files
- KWS_PREDICTIONS: folder for storing predicted outputs from KWS
- SIMILARITY_MATRICES: folder for storing similarity matrices for
    embeddings used in KWS. See `README.md` inside for more info.
- EMBEDDINGS: folder for storing embeddings used for KWS
"""

KWS_PREDICTIONS = DATA_DIR / "kws_predictions"
SIMILARITY_MATRICES = DATA_DIR / "similarity_matrix/"
EMBEDDINGS = DATA_DIR / "tira_kws" / "embeddings"

# model and inference constants
WFST_BATCH_SIZE = 1024
MAX_KEYWORD_STR = '$max'
MEAN_KEYWORD_STR = '$mean'
CALIBRATION_NUM_NEGATIVE = 50
CALIBRATION_NUM_POSITIVE = 10
CLAP_IS_AVAILABLE = find_spec('clap') is not None
SPEECHBRAIN_IS_AVAILABLE = find_spec('speechbrain') is not None
WAV2VEC_DOWNSAMPLE_FACTOR = 320
