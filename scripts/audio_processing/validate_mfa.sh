corpus_dir=$DATASETS/tira_mfa
dictionary=~/projects/tira_kws/data/labels/alignment/mfa_dict.txt
acoustic_model=english_mfa
output_dir=~/projects/tira_kws/data/labels/alignment/mfa_output/

mfa model download acoustic $acoustic_model
mfa validate $corpus_dir $dictionary $acoustic_model $output_dir
