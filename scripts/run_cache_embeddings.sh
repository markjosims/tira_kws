for window_size in 0.05 0.1 0.2 0.5 1.0 2.0; do
    echo "Caching embeddings with window size: $window_size"
    for encoder_size in tiny base small; do
        echo "  Using encoder size: $encoder_size"
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