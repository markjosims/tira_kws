"""
Based on code in zipa/zipa_ctc_inference.py and zipa/zipformer_crctc/
retrieved on 16 Feb 2026.
Commit: f96afe2842868bb1d3cea1efe191806fdcd3c955
"""

from tira_kws.constants import (
    ZIPA_DIR, ZIPA_SMALL_CTC, ZIPA_SENTENCEPIECE_MODEL,
    ICEFALL_MODULE,
)
import sys
import torch
from typing import Dict, Any
import sentencepiece

if str(ICEFALL_MODULE) not in sys.path:
    sys.path.append(str(ICEFALL_MODULE))

from icefall import AttributeDict

if str(ZIPA_DIR) not in sys.path:
    sys.path.append(str(ZIPA_DIR))

zipa_crctc = ZIPA_DIR / "zipformer_crctc"
zipa_transducer = ZIPA_DIR / "zipformer_transducer"

if zipa_crctc not in sys.path:
    sys.path.append(str(zipa_crctc))

if zipa_transducer not in sys.path:
    sys.path.append(str(zipa_transducer))

from zipa_ctc_inference import small_params as ctc_params
from zipa_transducer_inference import small_params as transducer_params
from zipformer_crctc.train import get_model as get_ctc_model
from zipformer_transducer.train import get_model as get_transducer_model

def load_zipa_small_crctc(params: AttributeDict = ctc_params) -> torch.nn.Module:
    return get_ctc_model(params)
    
def load_zipa_small_transducer(params: AttributeDict = transducer_params) -> torch.nn.Module:
    if 'bpe_model' not in params:
        params = add_default_bpe_params(params)

    return get_transducer_model(params)

def add_default_bpe_params(params: AttributeDict) -> AttributeDict:
    bpe_model = sentencepiece.SentencePieceProcessor()
    bpe_model.load(str(ZIPA_SENTENCEPIECE_MODEL))
    params.blank_id = bpe_model.piece_to_id("<blk>")
    params.sos_id = params.eos_id = bpe_model.piece_to_id("<sos/eos>")
    params.vocab_size = bpe_model.get_piece_size()
    return params


def main():
    model = load_zipa_small_crctc()



    model = load_zipa_small_transducer()
   
    # # Generate a dummy audio batch (1 sample of 2 seconds of silence)
    # sample_rate = 16000
    # dummy_audio = [torch.zeros(int(sample_rate * 2)),torch.zeros(int(sample_rate * 2)),torch.zeros(int(sample_rate * 2))]  # 2-second silent audio

    # # Run inference
    # output = model.inference(dummy_audio)
    # print("Predicted transcript:", output)

if __name__ == "__main__":
    main()
