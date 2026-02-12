"""
For a given Lhotse FeatureSet, compute the pairwise distance matrix between
all pairs of keyword / test phrase embeddings, then save as a list of matrices
in a .npz file under data/distance/$feature_name alongside a CSV file 'manifest.csv'
containing the  keyword and test phrase indices for each matrix.
"""
