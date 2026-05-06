import pandas as pd
from tira_kws.constants import (
    CAPSTONE_KEYWORDS,
    CAPSTONE_CLOSE_NEGATIVE_RECORDS,
    CAPSTONE_POSITIVE_RECORDS,
    CAPSTONE_NEGATIVE_RECORDS,
    CAPSTONE_KWS_WORDLIST,
)


def main():
    keyword_df = pd.read_csv(CAPSTONE_KEYWORDS)
    sentences_dfs = [
        pd.read_csv(filepath)
        for filepath in (
            CAPSTONE_CLOSE_NEGATIVE_RECORDS,
            CAPSTONE_NEGATIVE_RECORDS,
            CAPSTONE_POSITIVE_RECORDS,
        )
    ]

    words = set()
    keyword_df["keyword"].str.split().apply(words.update)

    for df in sentences_dfs:
        df["audionorm_sentence"].str.split().apply(words.update)

    with open(CAPSTONE_KWS_WORDLIST, "w") as f:
        f.write("\n".join(words))


if __name__ == "__main__":
    main()
