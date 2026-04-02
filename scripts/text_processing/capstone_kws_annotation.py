"""
Interface for selecting keywords and keyword sentences.
"""

import keyword
import streamlit as st
import rapidfuzz

from tira_kws.constants import (
    CAPSTONE_DIR,
    CAPSTONE_KEYWORDS,
    CAPSTONE_POSITIVE_RECORDS,
    CAPSTONE_CLOSE_NEGATIVE_RECORDS,
    CAPSTONE_NEGATIVE_RECORDS,
    RECORD_LIST_CSV,
)
import pandas as pd
from pathlib import Path
import logging
import sys
from unidecode import unidecode

logger = logging.getLogger(__file__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

CAPSTONE_DIR.mkdir(exist_ok=True)

# try to load each csv
# if the file does not exist, create blank dataframe with expected columns

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
        "comment",
    ],
    "negative_df": [
        "sentence_id",
        "translation",
        "original_sentence",
        "textnorm_sentence",
        "audionorm_sentence",
        "audio_quality",
        "comment",
    ],
}

df2file = {
    "keyword_df": CAPSTONE_KEYWORDS,
    "positive_df": CAPSTONE_POSITIVE_RECORDS,
    "close_negative_df": CAPSTONE_CLOSE_NEGATIVE_RECORDS,
    "negative_df": CAPSTONE_NEGATIVE_RECORDS,
    "record_df": RECORD_LIST_CSV,
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

    return df


def load_all_dataframes():
    for key in df2file.keys():
        if key in st.session_state:
            continue
        df = load_dataframe(key)
        st.session_state[key] = df


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


def put_row_to_dataframe(key: str, row: dict[str, str | int], row_id: int):
    df = st.session_state[key]
    df.loc[row_id] = row
    st.session_state[key] = df


@st.cache_data
def get_all_words(df: pd.DataFrame, text_col: str = "text") -> set[str]:
    words = set()
    df[text_col].str.split().apply(words.update)
    return words


def get_ith_keyword(i: int) -> str:
    keywords = st.session_state["keyword_df"]["keyword"].tolist()
    return keywords[i]


load_all_dataframes()


keyword_df_edited = st.data_editor(
    st.session_state["keyword_df"], num_rows="dynamic", key="keyword_df_edits"
)
st.session_state["keyword_df_edited"] = keyword_df_edited

st.button(
    "Save keywords",
    on_click=lambda: save_dataframe(
        df=st.session_state["keyword_df_edited"], key="keyword_df"
    ),
)

current_keyword_i = st.pills(
    "Keyword query",
    options=list(range(len(st.session_state["keyword_df"]["keyword"]))),
    format_func=get_ith_keyword,
    key="current_keyword_i",
)
current_keyword = None
if current_keyword_i is not None:
    current_keyword = get_ith_keyword(current_keyword_i)

search_column = st.selectbox(
    "Text column to use for searching:",
    options=["text", "fst_normalized", "unidecode_normalized"],
)

all_words = get_all_words(st.session_state["record_df"], text_col=search_column)

selected_records = st.session_state["record_df"]
if current_keyword:
    query = current_keyword
    if search_column == "unidecode_normalized":
        query = unidecode(current_keyword)
    top_words = rapidfuzz.process.extract(
        current_keyword,
        all_words,
        limit=20,
        scorer=rapidfuzz.distance.DamerauLevenshtein.distance,
    )
    logger.debug(top_words)
    selected_hit = st.selectbox("Fuzzy match", options=[hit[0] for hit in top_words])
    st.session_state.selected_hit = selected_hit
    selected_records = st.session_state["record_df"][
        st.session_state["record_df"][search_column].str.contains(selected_hit)
    ]


selection = st.dataframe(
    selected_records,
    on_select="rerun",
    selection_mode="single-row",
)

list_to_edit = st.selectbox(
    label="Data file to add rows to",
    options=["positive_df", "negative_df", "close_negative_df"],
    key="list_to_edit",
)

rows = None
if selection and "selection" in selection and "rows" in selection["selection"]:
    rows = selection["selection"]["rows"]
if current_keyword and rows and st.session_state.get("list_to_edit", None):
    with st.container(border=True):
        list_to_edit = st.session_state["list_to_edit"]
        row_index = rows[0]

        row_data = selected_records.iloc[row_index]
        st.markdown(f"**Unnormalized text**:\t{row_data['text']}")
        st.markdown(f"**Translation**:\t{row_data['translation']}")
        st.markdown(f"**Keyword**:\t\t{current_keyword}")
        textnorm_sentence = st.text_input(
            label="Updated sentence", value=row_data["fst_normalized"]
        )
        new_row = {
            "keyword": current_keyword,
            "keyword_id": current_keyword_i,
            "sentence_id": row_data["record_idx"],
            "original_sentence": row_data["text"],
            "textnorm_sentence": textnorm_sentence,
            "translation": row_data["translation"],
        }
        add_row = st.button(
            label=f"Add row to {list_to_edit}",
            on_click=lambda: put_row_to_dataframe(
                key=list_to_edit,
                row_id=row_index,
                row=new_row,
            ),
        )


if list_to_edit:
    st.data_editor(st.session_state[list_to_edit], num_rows="delete")

logger.debug(selection)
