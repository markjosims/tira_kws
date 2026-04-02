"""
Interface for validating transcriptions of KWS audio records
"""

import streamlit as st
from streamlit_advanced_audio import audix

from tira_kws.constants import RECORDS_DIR
from tira_kws.interface_utils import (
    load_all_dataframes,
    save_dataframe,
    put_row_to_dataframe,
)
from tira_kws.dataloading import load_elicitation_cuts
import pandas as pd
import numpy as np
import logging
import sys
from lhotse.cut import Cut


def get_all_sentence_ids() -> set[int]:
    sentence_ids = set()
    for df_key in ["positive_df", "negative_df", "close_negative_df"]:
        df = st.session_state[df_key]
        df["sentence_id"].apply(sentence_ids.add)
    return sentence_ids


def get_updated_row_count(df_key: str | None) -> tuple[int, int]:
    """
    Counts how many rows in specified dataframe have a non-NA value
    for "audionorm_sentence". Returns that count along with total
    number of rows in the dataframe.
    """
    if df_key is None:
        return 0, 0
    df = st.session_state[df_key]
    updated_rows = (~(df["audionorm_sentence"].isna())).sum()
    total_rows = len(df)
    return updated_rows, total_rows


@st.cache_data
def audio_cuts(index_list: list[int]):
    """
    Wraps `load_elicitation_cuts` so it can be decorated with `st.cache_data`
    """
    return load_elicitation_cuts(index_list)


@st.cache_data
def load_audio_for_record(record_id: int) -> tuple[np.ndarray, int]:
    cut: Cut = st.session_state["audio_cuts"].filter(
        lambda cut: cut.id == str(record_id)
    )
    if len(cut) == 0:
        raise ValueError(f"No cut found for record_id {record_id}")
    elif len(cut) > 1:
        raise ValueError(f"Multiple cuts found for record_id {record_id}")
    cut = cut[0]
    mono = cut.load_audio()[0]
    return mono, cut.sampling_rate


st.header("KWS Capstone audio validation")

# load dataframes, get sentence ids, then load respective audio
load_all_dataframes()
st.session_state["sentence_ids"] = get_all_sentence_ids()
st.session_state["audio_cuts"] = audio_cuts(st.session_state["sentence_ids"])


# pick which dataframe to annotate
st.divider()
list_to_edit = st.pills(
    "Select dataframe to validate:",
    options=["positive_df", "close_negative_df", "negative_df"],
)

# show progress bar
updated_rows, total_rows = get_updated_row_count(list_to_edit)
st.write(f"Progress validating {list_to_edit}: {updated_rows}/{total_rows}")
st.progress(updated_rows / total_rows if total_rows > 0 else 0)

if list_to_edit:
    selected_df = st.session_state[list_to_edit]
    event = st.dataframe(selected_df, selection_mode="single-row", on_select="rerun")

    # display dataframe, then when a row is selected make a container
    # for playing audio and updating sentence
    if (
        event
        and hasattr(event, "selection")
        and "rows" in event.selection
        and event["selection"]["rows"]
    ):
        row_index = event["selection"]["rows"][0]
        selected_row = selected_df.iloc[row_index]

        sentence_id = selected_row["sentence_id"]
        original_sentence = selected_row["textnorm_sentence"]
        translation = selected_row["translation"]
        keyword = selected_row["keyword"]

        with st.container(border=True):
            audio_array, sample_rate = load_audio_for_record(
                selected_row["sentence_id"]
            )
            st.subheader(f"Editing record {sentence_id}")
            st.markdown(f"**Keyword:** {keyword}")
            st.markdown(f"**Text:** {original_sentence}")
            st.markdown(f"**Translation:** {translation}")

            audix(audio_array, sample_rate=sample_rate)

            original_as_default = lambda: st.session_state.update(
                updated_sentence=original_sentence
            )
            st.button(
                "Copy original sentence",
                on_click=original_as_default
            )
            updated_sentence_default = st.session_state.get("updated_sentence", None)

            updated_sentence = st.text_input(
                label="Updated sentence",
                value=updated_sentence_default,
                help="Edit the sentence as needed to match the audio as accurately as possible.",
            )
            audio_quality = st.select_slider(
                label="Audio quality",
                options=range(1, 6),
                help="Rate the audio quality on a scale of 1 to 5, where 1 is hard to understand and 5 is excellent.",
                value=3,
            )
            extra_speech = st.checkbox(
                label="Extra speech in audio",
                value=False,
                help="Check if there is extra speech in the audio that is not captured in the sentence.",
            )
            missing_speech = st.checkbox(
                label="Missing speech in audio",
                value=False,
                help="Check if there is speech in the sentence that is missing from the audio.",
            )
            mistranscription = st.checkbox(
                label="Mistranscription",
                value=False,
                help="Check if the sentence is a mistranscription of the audio (e.g. completely different words, not just misspelling).",
            )
            row_data = {
                **selected_row,
                "audionorm_sentence": updated_sentence,
                "extra_speech": extra_speech,
                "missing_speech": missing_speech,
                "mistranscription": mistranscription,
                "audio_quality": audio_quality,
            }
            st.button(
                "Update Row",
                on_click=lambda: put_row_to_dataframe(
                    key=list_to_edit,
                    row=row_data,
                    row_id=row_index,
                ),
            )

    # button to save all data
    st.button(
        label=f"Save {list_to_edit}",
        on_click=lambda: save_dataframe(
            key=list_to_edit,
            df=st.session_state[list_to_edit],
        ),
    )
