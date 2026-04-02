"""
Interface for selecting keywords and keyword sentences.
"""

import streamlit as st
import rapidfuzz

from tira_kws.constants import (
    CAPSTONE_DIR,
    CAPSTONE_KWS_WORDLIST,
)
from tira_kws.interface_utils import (
    load_all_dataframes,
    save_dataframe,
    put_row_to_dataframe,
)
import pandas as pd
import logging
import sys
from unidecode import unidecode


logger = logging.getLogger(__file__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

CAPSTONE_DIR.mkdir(exist_ok=True)

@st.cache_data
def get_words_from_records(df: pd.DataFrame, text_col: str = "text") -> set[str]:
    words = set()
    df[text_col].str.split().apply(words.update)
    return words

def get_words_from_kws_data(col: str = "textnorm_sentence") -> set[str]:
    words = set()
    for df_key in ["positive_df", "negative_df", "close_negative_df"]:
        df = st.session_state[df_key]
        df[col].str.split().apply(words.update)
    return words

def save_words_from_kws_data(col: str = "textnorm_sentence"):
    kws_words = get_words_from_kws_data(col)
    with open(CAPSTONE_KWS_WORDLIST, 'w') as f:
        f.write('\n'.join(kws_words))
    st.toast(f"Saved wordlist from KWS data to {CAPSTONE_KWS_WORDLIST}")

def get_keyword_index(keyword_str: str) -> int:
    keywords = st.session_state["keyword_df"]["keyword"].tolist()
    return keywords.index(keyword_str)


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

keyword_list = st.session_state["keyword_df"]["keyword"]

current_keyword = st.pills(
    "Keyword query",
    options=st.session_state["keyword_df"]["keyword"],
    key="current_keyword",
)

query_non_keyword = st.checkbox(label="Query non-keyword")

query = current_keyword
if query_non_keyword:
    query = st.text_input(label="Query")
st.session_state.query = query

current_keyword_i = None
if current_keyword is not None:
    current_keyword_i = get_keyword_index(current_keyword)
st.session_state["current_keyword_i"] = current_keyword_i

with st.popover("Progress"):
    current_keyword = st.session_state.get("current_keyword", None)
    st.progress(
        value=len(keyword_list) / 10, text=f"Num keywords {len(keyword_list)}/10"
    )
    st.progress(
        value=len(st.session_state["negative_df"]) / 100,
        text=f"Negatives records: {len(st.session_state['negative_df'])}/100",
    )

    if current_keyword:
        keyword_positives = (
            st.session_state["positive_df"]["keyword"]
            .value_counts()
            .get(current_keyword, 0)
        )
        keyword_negatives = (
            st.session_state["close_negative_df"]["keyword"]
            .value_counts()
            .get(current_keyword, 0)
        )

        st.progress(
            value=keyword_positives / 10,
            text=f"Positive records for {current_keyword} {keyword_positives}/10",
        )
        st.progress(
            value=keyword_negatives / 10,
            text=f"Close negative records for {current_keyword} {keyword_negatives}/10",
        )

search_column = st.selectbox(
    "Text column to use for searching:",
    options=["text", "fst_normalized", "unidecode_normalized"],
)

all_words = get_words_from_records(st.session_state["record_df"], text_col=search_column)

selected_records = st.session_state["record_df"]
if st.session_state.get("query", None):
    query = st.session_state.query
    assert type(query) is str
    if search_column == "unidecode_normalized":
        query = unidecode(query)
    top_words = rapidfuzz.process.extract(
        query,
        all_words,
        limit=20,
        scorer=rapidfuzz.distance.DamerauLevenshtein.distance,
    )
    logger.debug(top_words)
    selected_hit = st.selectbox("Fuzzy match", options=[hit[0] for hit in top_words])
    st.session_state.selected_hit = selected_hit
    selected_records = st.session_state["record_df"][
        st.session_state["record_df"][search_column].str.contains(
            selected_hit, regex=True
        )
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
if (
    (current_keyword or list_to_edit == "negative_df")
    and rows
    and st.session_state.get("list_to_edit", None)
):
    with st.container(border=True):
        list_to_edit = st.session_state["list_to_edit"]
        row_index = rows[0]

        row_data = selected_records.iloc[row_index]
        st.markdown(f"**Unnormalized text**:\t{row_data['text']}")
        st.markdown(f"**Translation**:\t{row_data['translation']}")
        if list_to_edit != "negative_df":
            st.markdown(f"**Keyword**:\t\t{current_keyword}")

        textnorm_sentence_type = st.pills(
            "Sentence column to use:", options=["text", "fst_normalized"]
        )
        textnorm_sentence = row_data[textnorm_sentence_type or "text"]
        textnorm_sentence = st.text_input(
            label="Updated sentence", value=textnorm_sentence
        )
        new_row = {
            "sentence_id": row_data["record_idx"],
            "original_sentence": row_data["text"],
            "textnorm_sentence": textnorm_sentence,
            "translation": row_data["translation"],
        }
        if list_to_edit != "negative_df":
            new_row["keyword"] = current_keyword
            new_row["keyword_i"] = current_keyword_i
        add_row = st.button(
            label=f"Add row to {list_to_edit}",
            on_click=lambda: put_row_to_dataframe(
                key=list_to_edit,
                row=new_row,
            ),
        )


if list_to_edit:
    edited_df = st.data_editor(st.session_state[list_to_edit], num_rows="delete")
    st.button(
        label=f"Save {list_to_edit}",
        on_click=lambda: save_dataframe(
            key=list_to_edit,
            df=edited_df,
        ),
    )

st.button(
    label="Save all words in KWS data to list (for MFA alignment)",
    on_click=save_words_from_kws_data
)

logger.debug(selection)
