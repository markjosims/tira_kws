"""
Based on code in zipa/zipa_ctc_inference.py and zipa/zipformer_crctc/
retrieved on 16 Feb 2026.
Commit: f96afe2842868bb1d3cea1efe191806fdcd3c955

Usage for CTC output:
wav -> features???
features -> model.forward_encoder
encoder_out -> model.ctc_output
ctc_output -> add wildcard
ctc_output_w_wildcard -> torch.nn.functional.ctc_loss

"""

from tira_kws.constants import (
    ZIPA_DIR,
    ZIPA_SMALL_CTC,
    ZIPA_SENTENCEPIECE_MODEL,
    ICEFALL_MODULE,
    MODEL_DIR,
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
from zipa_ctc_inference import large_params as ctc_params_large
from zipa_ctc_inference import ZIPA_CTC
from zipa_transducer_inference import small_params as transducer_params
from zipa_transducer_inference import large_params as transducer_params_large

# from zipformer_crctc.train import get_model as get_ctc_model
from zipformer_transducer.train import get_model as get_transducer_model
from zipformer_crctc.ctc_decode import decode_one_batch as _decode_ctc
from zipformer_transducer.decode import decode_one_batch as _decode_transducer


def load_zipa_small_crctc(params: AttributeDict = ctc_params) -> torch.nn.Module:
    raise NotImplementedError("Don't use!")
    # return get_ctc_model(params)


def load_zipa_large_crctc(params: AttributeDict = ctc_params_large) -> torch.nn.Module:
    if "model_path" not in params:
        params.model_path = str(
            MODEL_DIR / "zipa" / "zipa_large_crctc_0.5_scale_800000_avg10.pth"
        )
    if "device" not in params:
        params.device = "cuda"
    if "bpe_model" not in params:
        params = add_default_bpe_params(params)

    # return get_ctc_model(ctc_params_large)
    return ZIPA_CTC(params)


def load_zipa_small_transducer(
    params: AttributeDict = transducer_params,
) -> torch.nn.Module:
    raise NotImplementedError("Don't use!")
    if "bpe_model" not in params:
        params = add_default_bpe_params(params)

    return get_transducer_model(params)


def load_zipa_large_transducer(
    params: AttributeDict = transducer_params_large,
) -> torch.nn.Module:
    raise NotImplementedError("Don't use!")
    if "bpe_model" not in params:
        params = add_default_bpe_params(params)

    return get_transducer_model(transducer_params_large)


def add_default_bpe_params(params: AttributeDict) -> AttributeDict:
    bpe_model = sentencepiece.SentencePieceProcessor()
    bpe_model.load(str(ZIPA_SENTENCEPIECE_MODEL))
    params.bpe_model = str(ZIPA_SENTENCEPIECE_MODEL)
    params.blank_id = bpe_model.piece_to_id("<blk>")
    params.sos_id = params.eos_id = bpe_model.piece_to_id("<sos/eos>")
    params.vocab_size = bpe_model.get_piece_size()
    return params


def decode_ctc(
    params: AttributeDict,
    model: torch.nn.Module,
    sp: sentencepiece.SentencePieceProcessor,
    batch: dict,
) -> dict:
    if not hasattr(params, "decoding_method"):
        params.decoding_method = "ctc-greedy-search"
    device = torch.device("cuda")
    dummy_H = torch.tensor(0, device=device)
    return _decode_ctc(
        params=params,
        model=model,
        bpe_model=sp,
        batch=batch,
        HLG=None,
        H=dummy_H,
        word_table=None,
        G=None,
    )


def decode_transducer(
    params: AttributeDict,
    model: torch.nn.Module,
    sp: sentencepiece.SentencePieceProcessor,
    batch: dict,
) -> dict:
    if not hasattr(params, "decoding_method"):
        params.decoding_method = "greedy_search"
    if not hasattr(params, "max_sym_per_frame"):
        params.max_sym_per_frame = 1
    return _decode_transducer(params=params, model=model, sp=sp, batch=batch)


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
