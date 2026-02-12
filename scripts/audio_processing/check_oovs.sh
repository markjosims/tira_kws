corpus_dir=$DATASETS/tira_mfa
dictionary=data/labels/alignment/mfa_dict.txt
output_dir=data/labels/alignment/mfa_output/

mfa find_oovs $corpus_dir $dictionary $output_dir 
