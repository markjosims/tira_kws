"""
Loads embeddings for in-domain and out-of-domain datasets and computes ROC AUC
for LID on embeddings from both datasets. Two strategies are supported for LID scoring:
1) Energy-based scoring: LID score is computed as the negative L2 norm of the embedding.
2) Logistic regression-based scoring: A logistic regression model is trained to
   distinguish between in-domain and out-of-domain embeddings, and the predicted
   probabilities are used as LID scores. $k$=5 cross-validation is used to evaluate ROC AUC.

Arguments:
    --in_domain -i: Name of dataset to use as in-domain embeddings.
    --out_domain -o: Name of dataset to use as out-of-domain embeddings.
    --strategy -t: Strategy to use for LID scoring. Choices are 'energy' and 'logreg'.
        Default: 'energy'
    --encoder -e: Encoder to use for generating embeddings. Default: 'clap_ipa'
    --encoder_size -s: Size of encoder to use. Default: 'base'
    --window_size -w: Size of sliding window (in seconds) to use for encoding. If not
        specified, no sliding window is used. Default: None
    --window_hop -p: Hop size (in seconds) to use for sliding window. If not specified,
        defaults to half of window_size. Default: None
"""




from encoding import add_sliding_window_args, add_encoder_args
from dataloading import add_dataset_optional_args
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scripts.cache_embeddings import load_embeddings
from argparse import ArgumentParser
import torch

def parse_args():
    parser = ArgumentParser(description="Compute LID ROC AUC")
    parser.add_argument(
        '--in_domain', '-i', type=str, default='tira_asr', help='In-domain dataset name.'
    )
    parser.add_argument(
        '--out_domain', '-o', type=str, default='tira_drz', help='Out-of-domain dataset name.'
    )
    parser.add_argument(
        '--strategy', '-t', type=str, default='energy', choices=['energy', 'logreg'],
    )
    parser.add_argument(
        '--no_cross_whiten', action='store_true',
        help='Disable cross-whitening embeddings. Instead compute z-score on in-domain and "\
            + "out-of-domain embeddings together.'
    )
    parser = add_encoder_args(parser)
    parser = add_sliding_window_args(parser)
    parser = add_dataset_optional_args(parser)
    return parser.parse_args()

def compute_energy_roc_auc(X, y):
    print("Computing energy-based ROC AUC...")
    scores = 1-X.norm(dim=1).cpu().numpy()
    roc_auc = roc_auc_score(y, scores)
    return roc_auc

def compute_logreg_roc_auc(X, y):
    k = 5
    print(f"Performing {k}-fold cross-validation for logistic regression...")
    logreg = LogisticRegression(max_iter=1000)
    scores = cross_val_score(
        logreg, X.cpu().numpy(), y, cv=k, scoring='roc_auc',
    )
    return scores.mean()

def main():
    args = parse_args()

    in_domain_argdict = vars(args).copy()
    out_domain_argdict = vars(args).copy()
    
    in_domain_argdict['dataset'] = args.in_domain
    out_domain_argdict['dataset'] = args.out_domain
    out_domain_argdict['whitened_from_dataset'] = args.in_domain

    in_domain_embed_dict = load_embeddings(in_domain_argdict)
    out_domain_embed_dict = load_embeddings(out_domain_argdict)

    if not args.no_cross_whiten:
        in_domain_embeds = in_domain_embed_dict['whitened']
        out_domain_embeds = out_domain_embed_dict['cross_whitened']

        X = torch.cat([in_domain_embeds, out_domain_embeds], dim=0)
        y = torch.cat([
            torch.zeros(in_domain_embeds.size(0)),
            torch.ones(out_domain_embeds.size(0))
        ], dim=0)
    else:
        in_domain_embeds = in_domain_embed_dict['embeddings']
        out_domain_embeds = out_domain_embed_dict['embeddings']
        X = torch.cat([in_domain_embeds, out_domain_embeds], dim=0)
        X = (X - X.mean(dim=0)) / X.std(dim=0)
        y = torch.cat([
            torch.zeros(in_domain_embeds.size(0)),
            torch.ones(out_domain_embeds.size(0))
        ], dim=0)

    print("In-domain dataset:", args.in_domain)
    print("Out-of-domain dataset:", args.out_domain)
    print("Window size:", args.window_size)
    print("Encoder:", args.encoder)
    print("Encoder size:", args.encoder_size)
    if args.strategy == 'energy':
        roc_auc = compute_energy_roc_auc(X, y)
    elif args.strategy == 'logreg':
        roc_auc = compute_logreg_roc_auc(X, y)
    print(f"LID ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()