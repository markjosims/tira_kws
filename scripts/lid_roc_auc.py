"""
Loads embeddings for in-domain and out-of-domain datasets and computes ROC AUC
for LID based on magnitude of whitened embeddings.

Arguments:
    --in_domain -i: Name of dataset to use as in-domain embeddings.
    --out_domain -o: Name of dataset to use as out-of-domain embeddings.
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
    parser = add_encoder_args(parser)
    parser = add_sliding_window_args(parser)
    parser = add_dataset_optional_args(parser)
    return parser.parse_args()

def compute_lid_roc_auc(in_domain_embeds: torch.Tensor, out_domain_embeds: torch.Tensor):
    all_embeds = torch.cat([in_domain_embeds, out_domain_embeds], dim=0)
    labels = [1]*len(in_domain_embeds) + [0]*len(out_domain_embeds)

    scores = 1-all_embeds.norm(dim=1).cpu().numpy()
    roc_auc = roc_auc_score(labels, scores)
    return roc_auc

def main():
    args = parse_args()

    in_domain_argdict = vars(args).copy()
    out_domain_argdict = vars(args).copy()
    
    in_domain_argdict['dataset'] = args.in_domain
    out_domain_argdict['dataset'] = args.out_domain
    out_domain_argdict['whitened_from_dataset'] = args.in_domain

    breakpoint()
    in_domain_embed_dict = load_embeddings(in_domain_argdict)
    out_domain_embed_dict = load_embeddings(out_domain_argdict)

    in_domain_embeds = in_domain_embed_dict['whitened']
    out_domain_embeds = out_domain_embed_dict['cross_whitened']

    roc_auc = compute_lid_roc_auc(in_domain_embeds, out_domain_embeds)
    print("In-domain dataset:", args.in_domain)
    print("Out-of-domain dataset:", args.out_domain)
    print("Window size:", args.window_size)
    print("Encoder:", args.encoder)
    print("Encoder size:", args.encoder_size)
    print(f"LID ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()