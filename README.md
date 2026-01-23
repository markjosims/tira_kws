# tira_kws
Code for Keyword and Keyphrase Search experiments for Tira.

## Interspeech 2026
Proof-of-concept of KWS effectiveness on Tira datset using CLAP-IPA and XLS-R.
Follow-up on Wisniewski et al. (2025).
Performs KWS with individual words as query against target sentences using DTW.

## Sparse annotation filling
This experiment is more complicated.
It will involve entire sentences as queries and compare embedding-based strategies with ASR-based, and will use WFST-based decoding with k2.