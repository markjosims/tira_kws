# randomly select $num_files recordings and alignments
# for visual inspection of alignment quality
num_files="${1:-10}"

corpus_dir=$DATASETS/tira_mfa
speaker_dir=$corpus_dir/himidan
output_dir=data/alignment/mfa_output/

mapfile -t textgrid_files < <(find $output_dir -type f -name "*.TextGrid" | shuf -n $num_files)
wav_files=()
for textgrid_file in "${textgrid_files[@]}"; do
    textgrid_basename=$(basename "$textgrid_file" .TextGrid)
    wav_file="$speaker_dir/$textgrid_basename.wav"
    if [[ -f "$wav_file" ]]; then
        wav_files+=("$wav_file")
    else
        echo "Warning: Corresponding WAV file not found for $textgrid_file at $wav_file"
    fi
done
praat --open ${wav_files[@]} ${textgrid_files[@]}