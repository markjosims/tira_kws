"""
Based on code in egs/gigaspeech/KWS/zipformer/decode-asr.py,
retrieved on 16 Feb 2026.
Commit: 0904e490c5fb424dc5cb4d14ae468e4d32a07dc4
"""

from dataclasses import dataclass, field
import math
import os
from typing import List, Optional, Set, Tuple, Dict

import k2
import torch
from torch import nn
import warnings
from icefall import ContextGraph, ContextState, NgramLmStateCost
from icefall.utils import (
    AttributeDict,
    store_transcripts,
    write_error_stats,
    KeywordResult,
)
import argparse

from tira_kws.constants import (
    KEYWORDS_CSV, ZIPA_SENTENCEPIECE_MODEL, DEVICE,
    ICEFALL_MODULE, GIGASPEECH_KWS_DIR, DISTANCE_DIR
)
from tira_kws.textproc import load_zipa_ipa_set, encode_sentencepiece
from tira_kws.models.zipa import load_zipa_small_transducer, transducer_params, add_default_bpe_params
from tira_kws.dataloading import get_k2_dataloader, load_kws_cuts
import pandas as pd
import sentencepiece
import sys

kws_decode_module = GIGASPEECH_KWS_DIR / "zipformer"

if str(ICEFALL_MODULE) not in sys.path:
    sys.path.append(str(ICEFALL_MODULE))
if str(kws_decode_module) not in sys.path:
    sys.path.append(str(kws_decode_module))

# copied from icefall/egs/gigaspeech/KWS/zipformer/decode.py
LOG_EPS = math.log(1e-10)

# following code copied from icefall/egs/gigaspeech/KWS/zipformer/beam_search.py

@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    ac_probs: Optional[List[float]] = None

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # the lm score for next token given the current ys
    lm_score: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # N-gram LM state
    state_cost: Optional[NgramLmStateCost] = None

    # Context graph state
    context_state: Optional[ContextState] = None

    num_tailing_blanks: int = 0

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.

        Caution:
          `self` is modified **in-place**.

        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)

def keywords_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    keywords_graph: ContextGraph,
    beam: int = 4,
    num_tailing_blanks: int = 0,
    blank_penalty: float = 0,
) -> List[List[KeywordResult]]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      keywords_graph:
        A instance of ContextGraph containing keywords and their configurations.
      beam:
        Number of active paths during the beam search.
      num_tailing_blanks:
        The number of tailing blanks a keyword should be followed, this is for the
        scenario that a keyword will be the prefix of another. In most cases, you
        can just set it to 0.
      blank_penalty:
        The score used to penalize blank probability.
    Returns:
      Return a list of list of KeywordResult.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert keywords_graph is not None

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                context_state=keywords_graph.root,
                timestamp=[],
                ac_probs=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    sorted_ans = [[] for _ in range(N)]
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]

        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        probs = logits.softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs = probs.log()

        probs = probs.reshape(-1)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
        ragged_probs = k2.RaggedTensor(shape=log_probs_shape, value=probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)
            hyp_probs = ragged_probs[i].tolist()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]
                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                new_ac_probs = hyp.ac_probs[:]
                context_score = 0
                new_context_state = hyp.context_state
                new_num_tailing_blanks = hyp.num_tailing_blanks + 1
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)
                    new_ac_probs.append(hyp_probs[topk_indexes[k]])
                    (
                        context_score,
                        new_context_state,
                        _,
                    ) = keywords_graph.forward_one_step(hyp.context_state, new_token)
                    new_num_tailing_blanks = 0
                    if new_context_state.token == -1:  # root
                        new_ys[-context_size:] = [-1] * (context_size - 1) + [blank_id]

                new_log_prob = topk_log_probs[k] + context_score

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
                    ac_probs=new_ac_probs,
                    context_state=new_context_state,
                    num_tailing_blanks=new_num_tailing_blanks,
                )
                B[i].add(new_hyp)

            top_hyp = B[i].get_most_probable(length_norm=True)
            matched, matched_state = keywords_graph.is_matched(top_hyp.context_state)
            if matched:
                ac_prob = (
                    sum(top_hyp.ac_probs[-matched_state.level :]) / matched_state.level
                )
            if (
                matched
                and top_hyp.num_tailing_blanks > num_tailing_blanks
                and ac_prob >= matched_state.ac_threshold
            ):
                keyword = KeywordResult(
                    hyps=top_hyp.ys[-matched_state.level :],
                    timestamps=top_hyp.timestamp[-matched_state.level :],
                    phrase=matched_state.phrase,
                )
                sorted_ans[i].append(keyword)
                B[i] = HypothesisList()
                B[i].add(
                    Hypothesis(
                        ys=[-1] * (context_size - 1) + [blank_id],
                        log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                        context_state=keywords_graph.root,
                        timestamp=[],
                        ac_probs=[],
                    )
                )

    B = B + finalized_B

    for i, hyps in enumerate(B):
        top_hyp = hyps.get_most_probable(length_norm=True)
        matched, matched_state = keywords_graph.is_matched(top_hyp.context_state)
        if matched:
            ac_prob = (
                sum(top_hyp.ac_probs[-matched_state.level :]) / matched_state.level
            )
        if matched and ac_prob >= matched_state.ac_threshold:
            keyword = KeywordResult(
                hyps=top_hyp.ys[-matched_state.level :],
                timestamps=top_hyp.timestamp[-matched_state.level :],
                phrase=matched_state.phrase,
            )
            sorted_ans[i].append(keyword)

    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans

def get_hyps_shape(hyps: List[HypothesisList]) -> k2.RaggedShape:
    """Return a ragged shape with axes [utt][num_hyps].

    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return a ragged shape with 2 axes [utt][num_hyps]. Note that
      the shape is on CPU.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    ans = k2.ragged.create_ragged_shape2(
        row_splits=row_splits, cached_tot_size=row_splits[-1].item()
    )
    return ans


# following functions copied from icefall/egs/gigaspeech/KWS/zipformer/decode.py

@dataclass
class KwMetric:
    TP: int = 0  # True positive
    FN: int = 0  # False negative
    FP: int = 0  # False positive
    TN: int = 0  # True negative
    FN_list: List[str] = field(default_factory=list)
    FP_list: List[str] = field(default_factory=list)
    TP_list: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"(TP:{self.TP}, FN:{self.FN}, FP:{self.FP}, TN:{self.TN})"

def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: sentencepiece.SentencePieceProcessor,
    batch: dict,
    keywords_graph: Optional[ContextGraph] = None,
) -> List[List[Tuple[str, Tuple[int, int]]]]:
    """Decode one batch and return the result in a list.

    The length of the list equals to batch size, the i-th element contains the
    triggered keywords for the i-th utterance in the given batch. The triggered
    keywords are also a list, each of it contains a tuple of hitting keyword and
    the corresponding start timestamps and end timestamps of the hitting keyword.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      keywords_graph:
        The graph containing keywords.
    Returns:
      Return the decoding result. See above description for the format of
      the returned list.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.causal:
        # this seems to cause insertions at the end of the utterance if used with zipformer.
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPS,
        )

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    ans_dict = keywords_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        keywords_graph=keywords_graph,
        beam=params.beam,
        num_tailing_blanks=params.num_tailing_blanks,
        blank_penalty=params.blank_penalty,
    )

    hyps = []
    for ans in ans_dict:
        hyp = []
        for hit in ans:
            hyp.append((hit.phrase, (hit.timestamps[0], hit.timestamps[-1])))
        hyps.append(hyp)

    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: sentencepiece.SentencePieceProcessor,
    keywords_graph: ContextGraph,
    keywords: Set[str],
    test_only_keywords: bool,
) -> Tuple[List[Tuple[str, List[str], List[str]]], KwMetric]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      keywords_graph:
        The graph containing keywords.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    log_interval = 50

    results = []
    metric = {"all": KwMetric()}
    for k in keywords:
        metric[k] = KwMetric()

    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            keywords_graph=keywords_graph,
            batch=batch,
        )

        this_batch = []
        assert len(hyps) == len(texts)
        for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
            ref_text = ref_text.upper()
            ref_words = ref_text.split()
            hyp_words = [x[0] for x in hyp_words]
            # for computing WER
            this_batch.append((cut_id, ref_words, " ".join(hyp_words).split()))
            hyp_set = set(hyp_words)  # each item is a keyword phrase
            if len(hyp_words) > 1:
                print(
                    f"Cut {cut_id} triggers more than one keywords : {hyp_words},"
                    f"please check the transcript to see if it really has more "
                    f"than one keywords, if so consider splitting this audio and"
                    f"keep only one keyword for each audio."
                )
            hyp_str = " | ".join(
                hyp_words
            )  # The triggered keywords for this utterance.
            TP = False
            FP = False
            for x in hyp_set:
                assert x in keywords, x  # can only trigger keywords
                if (test_only_keywords and x == ref_text) or (
                    not test_only_keywords and x in ref_text
                ):
                    TP = True
                    metric[x].TP += 1
                    metric[x].TP_list.append(f"({ref_text} -> {x})")
                if (test_only_keywords and x != ref_text) or (
                    not test_only_keywords and x not in ref_text
                ):
                    FP = True
                    metric[x].FP += 1
                    metric[x].FP_list.append(f"({ref_text} -> {x})")
            if TP:
                metric["all"].TP += 1
            if FP:
                metric["all"].FP += 1
            TN = True  # all keywords are true negative then the summery is true negative.
            FN = False
            for x in keywords:
                if x not in ref_text and x not in hyp_set:
                    metric[x].TN += 1
                    continue

                TN = False
                if (test_only_keywords and x == ref_text) or (
                    not test_only_keywords and x in ref_text
                ):
                    fn = True
                    for y in hyp_set:
                        if (test_only_keywords and y == ref_text) or (
                            not test_only_keywords and y in ref_text
                        ):
                            fn = False
                            break
                    if fn:
                        FN = True
                        metric[x].FN += 1
                        metric[x].FN_list.append(f"({ref_text} -> {hyp_str})")
            if TN:
                metric["all"].TN += 1
            if FN:
                metric["all"].FN += 1

        results.extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            print(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results, metric


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]],
    metric: KwMetric,
):
    recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
    results = sorted(results)
    store_transcripts(filename=recog_path, texts=results)
    print(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(f, f"{test_set_name}", results, enable_log=True)
    print("Wrote detailed error stats to {}".format(errs_filename))

    metric_filename = params.res_dir / f"metric-{test_set_name}-{params.suffix}.txt"

    with open(metric_filename, "w") as of:
        width = 10
        for key, item in sorted(
            metric.items(), key=lambda x: (x[1].FP, x[1].FN), reverse=True
        ):
            acc = (item.TP + item.TN) / (item.TP + item.TN + item.FP + item.FN)
            precision = (
                0.0 if (item.TP + item.FP) == 0 else item.TP / (item.TP + item.FP)
            )
            recall = 0.0 if (item.TP + item.FN) == 0 else item.TP / (item.TP + item.FN)
            fpr = 0.0 if (item.FP + item.TN) == 0 else item.FP / (item.FP + item.TN)
            s = f"{key}:\n"
            s += f"\t{'TP':{width}}{'FP':{width}}{'FN':{width}}{'TN':{width}}\n"
            s += f"\t{str(item.TP):{width}}{str(item.FP):{width}}{str(item.FN):{width}}{str(item.TN):{width}}\n"
            s += f"\tAccuracy: {acc:.3f}\n"
            s += f"\tPrecision: {precision:.3f}\n"
            s += f"\tRecall(PPR): {recall:.3f}\n"
            s += f"\tFPR: {fpr:.3f}\n"
            s += f"\tF1: {0.0 if precision * recall == 0 else 2 * precision * recall / (precision + recall):.3f}\n"
            if key != "all":
                s += f"\tTP list: {' # '.join(item.TP_list)}\n"
                s += f"\tFP list: {' # '.join(item.FP_list)}\n"
                s += f"\tFN list: {' # '.join(item.FN_list)}\n"
            of.write(s + "\n")
            if key == "all":
                print(s)
        of.write(f"\n\n{params.keywords_config}")

    print("Wrote metric stats to {}".format(metric_filename))

# end copied code

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading keywords and building context graph...")
    keywords_df = pd.read_csv(args.keywords_file, keep_default_na=False)
    keywords = keywords_df["word"].tolist()
    ipa_set = load_zipa_ipa_set()
    sentpiece_model = sentencepiece.SentencePieceProcessor(
        model_file=str(args.sentencepiece_model)
    )
    token_ids = keywords_df["word"].apply(
        lambda x: encode_sentencepiece(x, sentpiece_model, ipa_set)
    ).tolist()

    keywords_graph = ContextGraph(
        context_score=args.keywords_score, ac_threshold=args.keywords_threshold
    )
    keywords_graph.build(
        token_ids=token_ids,
        phrases=keywords,
    )
    keywords = set(keywords_df["word"].tolist())

    print("Loading ZIPA Transducer model...")
    params = AttributeDict(**transducer_params)
    params = add_default_bpe_params(params)
    model = load_zipa_small_transducer(params)
    model.to(DEVICE)
    model.eval()

    print("Loading test phrase cuts...")
    # ZIPA expects filterbank features
    _, test_phrase_cuts = load_kws_cuts(feature_set='fbank', normalize_text=True)
    test_phrase_cuts = test_phrase_cuts.to_eager()
    print(f"Loaded {len(test_phrase_cuts)} cuts for test phrases.")

    print("Building test phrase dataloader...")
    dataloader = get_k2_dataloader(test_phrase_cuts, args)

    print("Setting decoding parameters...")
    params.contextsize = args.contextsize
    params.keywords_score = args.keywords_score
    params.keywords_threshold = args.keywords_threshold
    params.beam = args.beam
    params.num_tailing_blanks = args.num_tailing_blanks
    params.blank_penalty = args.blank_penalty


    print("Decoding with context graph...")
    results, metric = decode_dataset(
        dl=dataloader,
        params=params,
        model=model,
        sp=sentpiece_model,
        keywords_graph=keywords_graph,
        keywords=keywords,
        # test_only_keywords is set to False since the 'text'
        # field of supervisions contains the full sentence, not just the keyword
        test_only_keywords=False,
    )
    print("Decoding complete.")

    # save results expects params to have following attrs:
    params.res_dir = args.output_dir
    params.suffix = "contextgraph"
    params.keywords_config = "_".join(keywords)

    save_results(
        params=params,
        test_set_name="tira_kws",
        results=results,
        metric=metric,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keywords_file",
        type=str,
        default=KEYWORDS_CSV,
    )
    parser.add_argument(
        "--sentencepiece_model",
        type=str,
        default=ZIPA_SENTENCEPIECE_MODEL,
        help="Path to sentencepiece model for tokenizing keywords."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DISTANCE_DIR/"zipa_ctc",
    )
    parser.add_argument(
        "--max_cuts",
        type=int,
        default=32,
        help="Maximum number of cuts per batch.",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=300.0,
        help="Maximum duration of a batch in seconds.",
    )

    # following args copied from icefall/egs/gigaspeech/KWS/zipformer/decode-asr.py
    parser.add_argument(
        "--contextsize",
        type=int,
        default=0, # ZIPA uses unigram model
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )

    parser.add_argument(
        "--keywords_score",
        type=float,
        default=1.5,
        help="""
        The default boosting score (token level) for keywords. it will boost the
        paths that match keywords to make them survive beam search.
        """,
    )

    parser.add_argument(
        "--keywords_threshold",
        type=float,
        default=0.35,
        help="The default threshold (probability) to trigger the keyword.",
    )

    parser.add_argument(
        "--beam",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--num_tailing_blanks",
        type=int,
        default=1,
        help="The number of tailing blanks should have after hitting one keyword.",
    )

    parser.add_argument(
        "--blank_penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    return parser.parse_args()

if __name__ == "__main__":
    main()