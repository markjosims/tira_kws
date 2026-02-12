"""
Load a pairwise distance matrix and manifest file computed by
compute_pairwise_distance.py and compute the DTW score for each
pair of keyword / test phrase, then update the manifest file with
the DTW score and save as a new CSV file 'manifest_dtw.csv' under
data/distance/$feature_name.
"""