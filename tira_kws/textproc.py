from tira_kws.constants import ZIPA_IPA_SET
import sentencepiece
from typing import List, Union

def load_zipa_ipa_set():
    with open(ZIPA_IPA_SET, "r") as f:
        ipa_set = set(line.strip() for line in f)
    return ipa_set

def remove_oov_chars(text, ipa_set):
    return "".join(char for char in text if char in ipa_set)

def encode_sentencepiece(
        text: str,
        sentpiece_model: Union[str, sentencepiece.SentencePieceProcessor],
        ipa_set: List[str] = None
):
    if type(sentpiece_model) is str:
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(str(sentpiece_model))
    else:
        sp = sentpiece_model
    if ipa_set is not None:
        text = remove_oov_chars(text, ipa_set)
    return sp.encode(text, out_type=int)