from typing import Dict, Optional, Sequence, Union
from unicodedata import normalize
from string import punctuation
from argparse import ArgumentParser
from src.constants import WORDS_CSV, MFA_DICT_PATH, ALIGNMENT_DIR
import pandas as pd
from tqdm import tqdm

TIRA2MFA = {
    # these characters are unchanged
    # still including to ensure proper spacing in MFA output
    "w": "w",
    "l": "l",
    "m": "m",
    "j": "j",
    "k": "k",
    "s": "s",
    "v": "v",
    "ʃ": "ʃ",
    "p": "p",
    "n": "n",
    "ə": "ə",
    "f": "f",
    "ð": "ð",
    "ɭ": "l",
    "ɾ": "ɾ",
    "ɛ": "ɛ",
    "h": "h",
    "t": "t",
    "b": "b",
    "ɲ": "ɲ",
    "d": "d",
    "ŋ": "ŋ",
    "ʊ": "ʊ",
    # non-vacuous replacements
    "o": "ow",
    "a": "\u0251", # IPA open back unrounded vowel
    "ɔ": "ɒ",
    "r": "ɹ",
    "ɽ": "ɹ",
    "c": "tʃ",
    "i": "i",
    "e": "ɛ",
    "g": "\u0261", # IPA [g]
    "u": "ʉ",
    "ɟ": "dʒ",
    "ɜ": "ɐ",
    "\u026a": "\u026a"
}

# add trailing space to each value
# since MFA will expect symbols to be separated by spaces

TIRA2MFA = {k: v+' ' for k, v in TIRA2MFA.items()}

DIACRITICS = {
    'grave': "\u0300",
    'macrn': "\u0304",
    'acute': "\u0301",
    'circm': "\u0302",
    'caron': "\u030C",
    'tilde': "\u0303",
    'bridge': "\u032A",
}

def tira2mfa(tira_str: str) -> str:
    # normalize unicode, remove diacritics and punctuation
    nfkd_norm = normalize('NFKD', tira_str)
    no_diac_str = strip_diacs(nfkd_norm)
    no_punct_str = remove_punct(no_diac_str)

    # now make replacements to get MFA format then trim trailing whitespace
    mfa_str = make_replacements(no_punct_str, TIRA2MFA)
    trimmed_whitespace = ' '.join(mfa_str.split())
    return trimmed_whitespace

def strip_diacs(text: str) -> str:
    for diac in DIACRITICS.values():
        text = text.replace(diac, '')
    return text

def remove_punct(text: str, keep: Optional[Union[str, Sequence[str]]] = None) -> str:
    for p in punctuation:
        if keep and p in keep:
            continue
        text = text.replace(p, '')
    return text

def make_replacements(text: str, reps: Dict[str, str]) -> str:
    """
    Makes all replacements specified by `reps`, a dict whose keys are intabs
    and values are outtabs to replace them.
    Avoids transitivity by first replacing intabs to a unique char not found in the original string.
    """
    max_ord_str = max_ord_in_str(text)
    max_ord_reps = max_ord_in_str(''.join([*reps.keys(), *reps.values()]))
    max_ord = max(max_ord_str, max_ord_reps)
    intab2unique = {
        k: chr(max_ord+i+1) for i, k in enumerate(reps.keys())
    }
    unique2outtab = {
        intab2unique[k]: v for k, v in reps.items()
    }

    # sort intabs so that longest sequences come first
    intabs = sorted(reps.keys(), key=len, reverse=True)

    for intab in intabs:
        sentinel = intab2unique[intab]
        text = text.replace(intab, sentinel)
    for sentinel, outtab in unique2outtab.items():
        text = text.replace(sentinel, outtab)
    return text

def max_ord_in_str(text: str) -> int:
    """
    Get the maximum Unicode code point (ordinal) of any character in the string `text`.
    """

    return max(ord(c) for c in text)

def main():
    args = get_args()

    ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading Tira words from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    with open(args.output_file, 'w') as f:
        for word in tqdm(df['word'].tolist(), desc="Converting words to MFA format"):
            mfa_word = tira2mfa(word)
            f.write(f"{word}\t{mfa_word}\n")
    print(f"Wrote MFA dictionary to {args.output_file}")

def get_args():
    parser = ArgumentParser(description="Convert Tira words to MFA format")
    parser.add_argument(
        "--input_file", "-i",
        default=WORDS_CSV,
        help="Path to input file containing Tira words"
    )
    parser.add_argument(
        "--output_file", "-o",
        default=MFA_DICT_PATH,
        help="Path to output file for MFA pronunciation dictionary"
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()