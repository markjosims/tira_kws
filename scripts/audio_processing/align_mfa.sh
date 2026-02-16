corpus_dir=$DATASETS/tira_mfa
dictionary=data/alignment/mfa_dict.txt
acoustic_model=english_mfa
output_dir=data/alignment/mfa_output

mfa align $corpus_dir $dictionary $acoustic_model $output_dir
