import torch
import os
from pathlib import Path

# audio and model constants

SAMPLE_RATE = 16_000
DEVICE = torch.device(0 if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))

# file paths

## dataset paths
DATA_DIR = os.environ.get("DATASETS", os.path.expanduser("~/datasets"))
TIRA_ASR_PATH = os.path.join(DATA_DIR, "tira_asr")
TIRA_DRZ_PATH = os.path.join(DATA_DIR, "tira_drz")

## local data paths
CSV_PATH = Path("data/kws_predictions/most_predicted_word.csv")
LABELS_DIR = Path("data/labels")
KEYPHRASE_PATH = LABELS_DIR/"tira_keyphrases.txt"
SIMILARITY_MATRIX_PATH = Path("data/similarity_matrix/similarity_matrix.pt")
EMBEDDINGS_DIR = Path("data/embeddings")

# keyword constants
MAX_KEYWORD_STR = '$max'
MEAN_KEYWORD_STR = '$mean'