
for window in 1.0 2.0 4.0 6.0; do
    echo python scripts/roc_auc.py --window_size $window
    python scripts/roc_auc.py --window_size $window
done