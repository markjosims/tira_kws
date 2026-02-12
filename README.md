# tira_kws
Code for Keyword and Keyphrase Search experiments for Tira.

## Interspeech 2026
Proof-of-concept of KWS effectiveness on Tira datset using ZIPA and XLS-R.
Follow-up on Wisniewski et al. (2025), which did KWS on Spanish audio using XLS-R.
Performs KWS with individual words as query against target sentences using DTW.

Divided into the following stages:

### Text pre-processing
Using elicitation transcripts, identify potential keywords as well as positive and negative sentences, subject to the following constraints:
- `num_keywords=30` keywords are sampled.
- Only one keyword is used per lemma. If *àpɾí* "boy" is a keyword don't also include *àpɾí-ɲá* "boy-ACC".
- Each keyword must contain at least 5 segments (consonants or vowels).
- For each keyword `phrase_count=10` positive phrases are sampled.
- Every positive sentence is positive for exactly one keyword. If *àpɾí* "boy" and *jə̀-və̀lɛ̀ð-ɔ́* "pulled" are both keywords then *àpɾí jə̀-və̀lɛ̀ð-ɔ́ ðáŋàlà* "the boy pulled the sheep" cannot be used as a sentence.
- Positive sentences *may* contain a word which is a different form of the lemma as some other keyword. If *àpɾí* "boy" and *jə̀-və̀lɛ̀ð-ɔ́* "pulled" are both keywords then *ðàŋàl ð-á àpɾí və́lɛ̀ð-à* "The sheep, the boy will pull it" can be used as a positive sentence for *àpɾí*.
- Every test sentence (positive or negative) has at least `min_words_per_sentence=4` words.
- `negative_phrase_count=700` negative sentences are sampled that contain none of the 30 keyowrds.
- Negative sentences *may* contain a word which is a different form of the same lemma as a keyword. If *àpɾí* "boy" is a keyword then a negative sentence is allowed to contain *àpɾí-ɲá* "boy-ACC".  


Text pre-processing is handled by the following two files:
- `scripts/text_processing/get_words_and_phrases.py`: Retrieve word and phrases from Tira ASR dataset. Makes following files:
  - `data/labels/phrases/keyphrases_rewritten_merges.csv`: Mapping of hand-transcribed sentences to FST-normalized strings.
  - `data/labels/phrases/tira_phrases.csv`: Spreadsheet containing data about each unique phrase (w/ FST normalization) in the dataset, including token counts, gloss, lemmata.
  - `data/labels/phrases/record2phrase.csv`: Mapping of record ids (from elicitation supervisions) to unique phrases.
  - `data/labels/words/tira_words.csv`: Spreadsheet containing data about each unique word, including it's lemma form and token/phrase counts.
  - `data/labels/word2phrase.csv`: A many-to-many mapping of all words to all phrases containing them.
- `scripts/text_processing/keyword_list_builder.py`: Generate keyword and test phrase lists from words and phrases
  - `data/labels/keywords/keyword_list.json`: A JSON list of objects, one for each keyword, containing positive record/phrase indices for that keyword.
  - `data/labels/keywords/keyword_sentence_ids.csv`: A spreadsheet containing all supervision records used in KWS along with whether that record is positive or negative, and if positive, for what keyword.
  - `data/labels/keywords.csv`: A spreadsheet listing all words used as keywords. Strict subset of `tira_words.csv` containing the same columns.

### Keyword segmentation
Apply forced alignment to sentence transcripts using Montreal Forced Aligner (MFA), then segment out audio slices for keywords.
Uses `english_mfa` acoustic model for alignment.
- `scripts/text_processing/mfa_dict_builder.py`: Generate dictionary file for forced alignment with MFA, stored in `data/alignment/mfa_dict.txt`.
- `scripts/audio_processing/prepare_mfa_audio.py`: Make directory of sentences and transcriptions for aligning with MFA. Stored by default in `~/datasets/mfa/`.
- `scripts/audio_processing/check_oovs.sh`: Check if any out-of-vocabulary items are found in the dataset and, if so, save OOV lists to `data/labels/alignment/mfa_output`.
- `scripts/audio_processing/validate_mfa.sh`: Run `mfa validate` on corpus and dictionary.
- `scripts/audio_processing/align_mfa.sh`: Run `mfa align` on corpus, saving TextGrid files to `data/labels/alignment/mfa_output`.
- `scripts/audio_processing/make_keyword_cuts.py`: Create Lhotse supervision set for keywords and test phrases, using word-level alignments from MFA for keywords. By default saves to `~/datasets/tira_elicitation`

### Embedding generation
Not handled directly in this project but in `tira_databuilder`.
Generate embeddings for keywords and test phrases using XLS-R, ZIPA-CTC and ZIPA-Transducer and save to disk.

### DTW scoring
Getting DTW scores is handled in two stages, pre-computation of distance matrices and DTW scoring.

1. Generate pairwise distance matrices for all keywords and test phrases.
  - Save list of $300\times700$ Numpy matrices of size $M\times{N}$ to `data/distance/$feature_name/distances.pkl` for each query/test pair.
  - Save CSV to `data/distance/$feature_name/manifest.csv` containing columns 'index' (matrix index in `distances.pkl`), 'query_index' (ID for keyword), 'test_index' (ID for test phrase)
2. Do batched DTW on the list of distance matrices, appending a column 'dtw_score' to `manifest.csv` with the DTW distance between each pair.

### Evaluation
Load DTW scores from a `manifest.csv` file in `data/distance/$feature_name` and, using the positive/negative indices in `keyword_sentence_ids.csv`, compute F1, AUROC, ... [WIP]

## Sparse annotation filling
This experiment is more complicated.
It will involve entire sentences as queries and compare embedding-based strategies with ASR-based, and will use WFST-based decoding with k2.