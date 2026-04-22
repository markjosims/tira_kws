"""
Logic shared across audio and text annotation interfaces
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from tira_kws.constants import (
    CAPSTONE_CLOSE_NEGATIVE_RECORDS,
    CAPSTONE_KEYWORDS,
    CAPSTONE_NEGATIVE_RECORDS,
    CAPSTONE_POSITIVE_RECORDS,
    CAPSTONE_DIR,
    RECORD_LIST_CSV,
)


df2file = {
    "keyword_df": CAPSTONE_KEYWORDS,
    "positive_df": CAPSTONE_POSITIVE_RECORDS,
    "close_negative_df": CAPSTONE_CLOSE_NEGATIVE_RECORDS,
    "negative_df": CAPSTONE_NEGATIVE_RECORDS,
    "record_df": RECORD_LIST_CSV,
}

df2columns = {
    "keyword_df": ["keyword", "keyword_id", "gloss"],
    "positive_df": [
        "keyword",
        "keyword_id",
        "sentence_id",
        "translation",
        "original_sentence",
        "textnorm_sentence",
        "audionorm_sentence",
        "audio_quality",
        "extra_speech",
        "missing_speech",
        "mistranscription",
        "comment",
    ],
    "close_negative_df": [
        "keyword",
        "keyword_id",
        "sentence_id",
        "translation",
        "original_sentence",
        "textnorm_sentence",
        "audionorm_sentence",
        "audio_quality",
        "extra_speech",
        "missing_speech",
        "mistranscription",
        "comment",
    ],
    "negative_df": [
        "sentence_id",
        "translation",
        "original_sentence",
        "textnorm_sentence",
        "audionorm_sentence",
        "audio_quality",
        "extra_speech",
        "missing_speech",
        "mistranscription",
        "comment",
    ],
}


def load_dataframe(key: str) -> pd.DataFrame:
    filepath = df2file[key]

    # if a dataframe is listed in `df2columns`, it should be initialized lazily
    # if not, we expect it already exists
    create_if_not_found = key in df2columns

    if not filepath.exists() and create_if_not_found:
        columns = pd.Index(df2columns[key])
        df = pd.DataFrame(columns=columns)
    elif not filepath.exists():
        raise FileNotFoundError(filepath)
    else:
        df = pd.read_csv(str(filepath))
        st.toast(f"Read dataframe {key} from {filepath}")

    return df


def load_all_dataframes(cache_on_first_load=True):
    for key in df2file.keys():
        if key in st.session_state:
            continue
        df = load_dataframe(key)
        st.session_state[key] = df
    if cache_on_first_load and st.session_state.get("first_load", True):
        cache_all_dataframes()
        st.session_state["first_load"] = False


def cache_all_dataframes():
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    cache_dir = CAPSTONE_DIR / ".cache" / time_string
    cache_dir.mkdir(parents=True, exist_ok=True)

    for key, filepath in df2file.items():
        df: pd.DataFrame = st.session_state[key]
        cached_path = cache_dir / filepath.name
        df.to_csv(str(cached_path), index=False)
    st.toast(f"Cached dataframes to {cache_dir}")


def update_dataframe(
    df: pd.DataFrame | dict[str, pd.DataFrame], key: str | None = None
):
    """
    Updates the in-memory dataframe stored in the streamlit session state.
    Does NOT update the csv on disk.
    """
    if type(key) is str:
        assert type(df) is pd.DataFrame
        st.session_state[key] = df
        return

    assert type(df) is dict
    for key in df2file.keys():
        st.session_state[key] = df[key]


def save_dataframe(df: pd.DataFrame | dict[str, pd.DataFrame], key: str | None = None):
    """
    Updates in-memory dataframes and writes to disk.
    """
    update_dataframe(df=df, key=key)
    if type(key) is str:
        assert type(df) is pd.DataFrame
        filepath = df2file[key]
        file_str = str(filepath)
        df.to_csv(file_str, index=False)
        st.toast(f"Saved {key} to {file_str}!")
        return

    assert type(df) is dict
    for (
        item_key,
        item_df,
    ) in df.items():
        save_dataframe(df=item_df, key=item_key)


def put_row_to_dataframe(
    key: str, row: dict[str, str | int], row_id: int | None = None
):
    df = st.session_state[key]
    if row_id is None:
        df.loc[len(df)] = row
    else:
        df.loc[row_id] = row
    st.session_state[key] = df

