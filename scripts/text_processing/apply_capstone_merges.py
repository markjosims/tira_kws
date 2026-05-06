from tira_kws.constants import (
    CAPSTONE_POSITIVE_RECORDS,
    CAPSTONE_NEGATIVE_RECORDS,
    CAPSTONE_CLOSE_NEGATIVE_RECORDS,
    CAPSTONE_KEYWORDS,
    CAPSTONE_KWS_WORD_MERGES,
)
import pandas as pd


def main():
    # read in data files
    keyword_df = pd.read_csv(CAPSTONE_KEYWORDS)
    sentence_dfs = {
        filepath: pd.read_csv(filepath)
        for filepath in (
            CAPSTONE_POSITIVE_RECORDS,
            CAPSTONE_NEGATIVE_RECORDS,
            CAPSTONE_CLOSE_NEGATIVE_RECORDS,
        )
    }

    merge2word = {}
    with open(CAPSTONE_KWS_WORD_MERGES, "r") as f:
        for line in f.readlines():
            parts = line.split()
            anchor = parts[0]
            merges = parts[1:]
            for merge in merges:
                merge2word[merge] = anchor

    # sanity check: the 'anchor' for each keyword should be itself
    for keyword in keyword_df["keyword"].tolist():
        assert keyword == merge2word[keyword]

    def update_sentence(sentence: str) -> str:
        """
        Replace each word in sentence with anchor word
        from merges file.
        """
        new_sentence = " ".join([merge2word[word] for word in sentence.split()])
        return new_sentence

    for csv_path, df in sentence_dfs.items():
        df["final_sentence"] = df["audionorm_sentence"].apply(update_sentence)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
