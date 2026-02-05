# tira_kws
Code for Keyword and Keyphrase Search experiments for Tira.

## Interspeech 2026
Proof-of-concept of KWS effectiveness on Tira datset using CLAP-IPA and XLS-R.
Follow-up on Wisniewski et al. (2025).
Performs KWS with individual words as query against target sentences using DTW.  

Code sequence:
- `scripts/text_processing/get_words_and_phrases.py`: Retrieve word and phrases from Tira ASR dataset
- `scripts/text_processing/keyword_list_builder.py`: Generate keyword and test phrase lists from words and phrases
- `scripts/text_processing/mfa_dict_builder.py`: Generate dictionary file for forced alignment with MFA
- `scripts/audio_processing/prepare_mfa_audio.py`: Make directory of sentences and transcriptions for aligning with MFA.


## Sparse annotation filling
This experiment is more complicated.
It will involve entire sentences as queries and compare embedding-based strategies with ASR-based, and will use WFST-based decoding with k2.