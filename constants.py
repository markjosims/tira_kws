import torch
import os

# audio and model constants

SAMPLE_RATE = 16_000
DEVICE = torch.device(0 if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))

# file paths

DATA_DIR = os.environ.get("DATASETS", os.path.expanduser("~/datasets"))
TIRA_ASR_PATH = os.path.join(DATA_DIR, "tira_asr")
TIRA_DRZ_PATH = os.path.join(DATA_DIR, "tira_drz")