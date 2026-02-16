corpus_dir=$DATASETS/tira_mfa
dictionary=data/alignment/mfa_dict.txt
output_dir=data/alignment/mfa_output/

mfa find_oovs $corpus_dir $dictionary $output_dir 
