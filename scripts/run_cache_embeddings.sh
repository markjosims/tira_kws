
for encoder_size in base; do
    echo "Caching embeddings encoder size: $encoder_size"
    echo "  No windowing"
    echo "    Caching TIRA ASR dataset..."
    python -m scripts.cache_embeddings \
        --dataset tira_asr \
        --encoder_size $encoder_size
    echo "    Caching TIRA DRZ dataset (z-scored to TIRA ASR)..."
    python -m scripts.cache_embeddings \
        --dataset tira_drz \
        --encoder_size $encoder_size \
        --whitened_from_dataset tira_asr
    for window_size in 0.1 1.0 2.0; do
        echo "  using window size: $window_size"
        echo "    Caching TIRA ASR dataset..."
        python -m scripts.cache_embeddings \
            --dataset tira_asr \
            --window_size $window_size \
            --encoder_size $encoder_size
        echo "    Caching TIRA DRZ dataset (z-scored to TIRA ASR)..."
        python -m scripts.cache_embeddings \
            --dataset tira_drz \
            --window_size $window_size \
            --encoder_size $encoder_size \
            --whitened_from_dataset tira_asr
    done
done