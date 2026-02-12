corpus_dir=$DATASETS/tira_mfa
dictionary=data/labels/alignment/mfa_dict.txt
acoustic_model=english_mfa
output_dir=data/labels/alignment/mfa_output/

mfa model download acoustic $acoustic_model
mfa validate $corpus_dir $dictionary $acoustic_model $output_dir --clean
